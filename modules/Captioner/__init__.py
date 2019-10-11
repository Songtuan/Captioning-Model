import torch
import torch.nn as nn
import allennlp.nn.beam_search as allen_beam_search

from functools import partial


class Captioner(nn.Module):
    def __init__(self):
        super(Captioner, self).__init__()

    def _step(self, image_features, tokens, states):
        '''
        Implement single decode step
        :param image_features(torch.Tensor): image features produced by encoder,
        a tensor with shape (batch_size, num_boxes, feature_size)
        :param tokens(torch.Tensor): input tokens, a tensor with shape (batch_size)
        :param states(Dict[str, torch.Tensor]): a dict contains previous hidden state
        :return: a tuple (torch.Tensorm Dict[str, torch.Tensor])
        '''
        if image_features.shape[0] != tokens.shape[0]:
            image_features = image_features.unsqueeze(1).repeat(1, self.beam_size, 1, 1)
            batch_size, beam_size, num_boxes, image_feature_size = image_features.shape
            image_features = image_features.view(batch_size * beam_size, num_boxes, image_feature_size)

        token_embeddings = self._embedding_layer(tokens)
        logits, states = self._cell(image_features, token_embeddings, states)
        logits = self._output_layer(logits)
        log_probs = self._log_softmax(logits)

        if self.training:
            # in training mode, we need logits to calculate loss
            return logits, states
        else:
            # in eval mode, we need log_probs distribution of words
            return log_probs, states

    def forward(self, image_features, targets=None):
        '''
        Implement forward propagation
        :param image_features(torch.Tensor): image features produced by encoder, a tensor
        with shape (batch_size, num_boxes, feature_size)
        :param targets(torch.Tensor): ground-true captions, a tensor with shape (batch_size, max_length)
        :return:
        '''
        output = {}
        batch_size = image_features.shape[0]
        states = None

        if self.training:
            # in training mode, ground-true targets should not be None
            assert targets is not None
            # max decoder step we need to perform
            max_step = self.seq_length - 1
            # a tensor contains logits of each step
            logits_seq = torch.zeros(max_step, batch_size, len(self.vocab))

            for t in range(max_step):
                # perform decode step
                tokens = targets[:, t]
                # logits should has shape (batch_size, vocab_size)
                logits, states = self._step(image_features=image_features, tokens=tokens, states=states)
                # update logits_seq
                logits_seq[t] = logits

            # the ground-true targets should exclude the first token
            # '<start>' since out model do not produce this token at
            # the beginning of sequence
            gt = targets[1:]
            loss = self.criterion(logits_seq, gt)

            # add loss to output dict
            output['loss'] = loss
        else:
            end_index = self.vocab['<boundary>'] if '<boundary>' in self.vocab else self.vocab['<end>']
            start_index = self.vocab['<boundary>'] if '<boundary>' in self.vocab else self.vocab['<start>']
            beam_search = allen_beam_search.BeamSearch(end_index=end_index,
                                                       max_steps=self.seq_length, beam_size=self.beam_size,
                                                       per_node_beam_size=self.beam_size)
            init_tokens = torch.tensor([start_index]).expand(batch_size).cuda()
            step = partial(self._decode_step, image_features=image_features)
            top_k_preds, log_probs = beam_search.search(start_predictions=init_tokens, start_state=states, step=step)
            preds = top_k_preds[:, 0, :]
            output['seq'] = preds

        return output
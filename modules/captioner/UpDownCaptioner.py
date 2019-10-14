import torch
import torch.nn as nn

from modules.updown_cell import UpDownCell
from modules.captioner import Captioner


class UpDownCaptioner(Captioner):
    def __init__(self, vocab, image_feature_size=2048, embedding_size=1000, hidden_size=512,
                 attention_projection_size=512, seq_length=20, beam_size=3,
                 pretrained_embedding=None, state_machine=None):
        super(UpDownCaptioner, self).__init__()

        vocab_size = len(vocab)
        self.vocab = vocab
        self.seq_length = seq_length
        self.state_machine = state_machine
        self.image_feature_size = image_feature_size
        self.beam_size = beam_size

        # define up-down cell
        self._cell = UpDownCell(image_feature_size=image_feature_size, embedding_size=embedding_size,
                                hidden_size=hidden_size, attention_projection_size=attention_projection_size)
        # define embedding layer
        if pretrained_embedding is not None:
            # if use pre-trained word embedding
            self._embedding_layer = nn.Embedding.from_pretrained(pretrained_embedding).float()
        else:
            self._embedding_layer = nn.Embedding(num_embeddings=vocab_size,
                                                 embedding_dim=embedding_size)

        # produce the logits which used to soft-max distribution
        self._output_layer = nn.Linear(hidden_size, vocab_size, bias=True)
        self._log_softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.vocab['<unk>'])

    def load(self, PATH):
        self.load_state_dict(torch.load(PATH))

import unittest
import torch

from allennlp.training.metrics import BLEU
from allennlp.data import Vocabulary

from dataset import CaptionDataset
from modules.faster_rcnn import MaskRCNN_Benchmark
from modules.captioner.UpDownCaptioner import UpDownCaptioner
from models.CaptioningModel import CaptioningModel

from torch import optim
from torch.nn.utils import clip_grad_norm_

from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.config import cfg


def eval_batch(batch_data, model, evaluator):
    """
    perform evaluation over one batch data
    :param batch_data: Dict[str, torch.Tensor], batch data read from data loader
    :param model: Captioning model
    :evaluator: allennlp.training.metrics.BLEU, evaluate bleu_4 score
    :return:
    """
    model.eval()

    # get evaluation images and ground-true target
    images = batch_data['images']
    targets = batch_data['captions']

    output = model(images)
    predicts = output['seq']
    print(predicts.tolist())
    evaluator(predictions=predicts, gold_targets=targets)


def train_batch(batch_data, model, optimizer):
    """
    perform training on one single batch data
    :param batch_data: Dict[str, torch.Tensor], batch data read from data-loader
    :param model: Captioning model
    :param optimizer: torch.nn.optimizer
    :return: torch.Tensor loss in this batch
    """
    # get training images and ground-true target
    images = batch_data['images']
    targets = batch_data['captions']

    # force model in training mode
    # and clean the grads of model
    model.train()
    model.zero_grad()
    # get the output of captioning model
    # and make sure 'loss' is in the return dict
    output = model(images, targets)
    assert 'loss' in output

    # perform back-propagation
    # and clip the grads
    batch_loss = output['loss']
    batch_loss.backward()
    clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return batch_loss


class MyTestCase(unittest.TestCase):
    def test_train(self):
        vocab = {'<boundary>': 0, '<unk>': 1, 'have': 2, 'to': 3, 'do': 4, 'it': 5, 'go': 6, 'as': 7}
        fake_img = torch.rand(1, 3, 255, 255)
        fake_target = torch.tensor([[0, 5, 6, 3, 6, 2, 0]]).long()
        data = {'images': fake_img, 'captions': fake_target}
        encoder = MaskRCNN_Benchmark()
        decoder = UpDownCaptioner(vocab=vocab)
        model = CaptioningModel(encoder=encoder, captioner=decoder)
        optimizer = optim.Adam(params=model.parameters(), lr=4e-4)

        for epoch in range(7):
            loss = train_batch(data, model, optimizer)

            evaluator = BLEU(exclude_indices={vocab['<unk>'], vocab['<boundary>']})
            with torch.no_grad():
                eval_batch(data, model, evaluator)
            bleu_score = evaluator.get_metric()['BLEU']

            print('Epoch: {0:2d} | Epoch Loss: {1:7.3f} | BLEU_4 Score: {2:5.2f}'.format(epoch, loss, bleu_score))

        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()

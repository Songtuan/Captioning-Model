import os
import argparse

from torch import optim
from tqdm import tqdm
from allennlp.training.metrics import BLEU
from allennlp.data import Vocabulary
from dataset import CaptionDataset
from torch.utils.data import DataLoader
from modules.faster_rcnn import MaskRCNN_Benchmark
from modules.captioner.UpDownCaptioner import UpDownCaptioner
from models.CaptioningModel import CaptioningModel
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.config import cfg


def train_batch(batch_data, model, optimizer):
    """
    perform training on one single batch data
    :param batch_data: Dict[str, torch.Tensor], batch data read from data-loader
    :param model: Captioning model
    :param optimizer: torch.nn.optimizer
    :return:
    """
    # get training images and ground-true target
    images = batch_data['images']
    targets = batch_data['captions']
    # convert input images to suitable format required by maskrcnn_benchmark
    images = to_image_list(images, cfg.DATALOADER.SIZE_DIVISIBILITY)

    # force model in training mode
    # and clean the grads of model
    model.train()
    model.zero_grad()
    # get the output of captioning model
    # and make sure 'loss' is in the return dict
    output = model(images, targets)
    assert 'loss' in output

    # perform back-propagation
    loss = output['loss']
    loss.backward()
    optimizer.step()


parser = argparse.ArgumentParser('Training Parameters')
parser.add_argument('--epochs', type=int, default=30, help='setting the number of training epochs')
parser.add_argument('--lr', type=float, default=4e-4, help='setting the initial learning rate')
parser.add_argument('--batch_size', type=int, default=50, help='setting the batch size')
parser.add_argument('--check_point', default='UpDownCaptioner.pth', help='check point path')


if __name__ == '__main__':
    opt = parser.parse_args()
    epochs = opt.epochs
    lr = opt.lr
    batch_size = opt.batch_size
    check_point = opt.check_point

    # load vocabulary
    vocabulary_path = 'vocab/vocabulary'
    vocabulary = Vocabulary.from_files(vocabulary_path)
    vocab = vocabulary.get_token_to_index_vocabulary()

    # load training set
    training_set_path = os.path.join('data', 'TRAIN.hdf5')
    training_set = CaptionDataset(training_set_path)

    # load eval set
    eval_set_path = os.path.join('data', 'VAL.hdf5')
    eval_set = CaptionDataset(eval_set_path)

    # build data-loaders for both training set and eval set
    # make both of them iterable
    training_loader = DataLoader(dataset=training_set, batch_size=batch_size)
    eval_loader = DataLoader(dataset=eval_set, batch_size=batch_size)

    # build encoder
    encoder = MaskRCNN_Benchmark()

    # build decoder
    decoder = UpDownCaptioner(vocab=vocab)
    decoder.load(check_point)

    # build complete model
    model = CaptioningModel(encoder=encoder, captioner=decoder)

    optimizer = optim.Adam(params=model.parameters(), lr=lr)

    for epoch in tqdm(range(epochs)):
        for batch_data in tqdm(training_loader):
           train_batch(batch_data=batch_data, model=model)




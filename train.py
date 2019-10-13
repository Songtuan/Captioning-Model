import os
import argparse
import torch
import torch.nn as nn

from allennlp.training.metrics import BLEU
from dataset import CaptionDataset
from torch.utils.data import DataLoader


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


if __name__ == '__main__':
    opt = parser.parse_args()
    epochs = opt.epochs
    lr = opt.lr
    batch_size = opt.batch_size

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


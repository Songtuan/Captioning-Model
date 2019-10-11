import torch
import torch.nn as nn

from modules.updown_cell import UpDownCell
from modules.BeamSearch import BeamSearch
from functools import partial


class UpDownCaptioner(nn.Module):
    def __init__(self, vocab, image_feature_size=2048, embedding_size=512, hidden_size=512,
            attention_projection_size=512, seq_length=20, beam_size=3, 
            pretrained_embedding=None, state_machine=None):

            self.vocab = vocab
            self.state_machine = state_machine
            self.image_feature_size = image_feature_size

            if pretrained_embedding is not None:
                self._embedding_layer = nn.Embedding.from_pretrained(pretrained_embedding).float()
            else:
                self._embedding_layer = nn.Embedding(num_embeddings=len(self.vocab), 
                                                     embedding_dim=embedding_size)
                                        
            



        



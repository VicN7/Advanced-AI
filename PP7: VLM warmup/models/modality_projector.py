import torch.nn as nn


class ModalityProjector(nn.Module):
    def __init__(self, vision_hidden_dim, language_hidden_dim):
        super().__init__()
        self.vision_hidden_dim = vision_hidden_dim
        self.language_hidden_dim = language_hidden_dim
        self.proj = nn.Linear(vision_hidden_dim, language_hidden_dim, bias=False)

    def forward(self, x):
        return self.proj(x)

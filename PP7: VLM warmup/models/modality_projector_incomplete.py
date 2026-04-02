import torch.nn as nn


class ModalityProjector(nn.Module):
    def __init__(self, vision_hidden_dim, language_hidden_dim, scale_factor):
        super().__init__()
        self.vision_hidden_dim = vision_hidden_dim
        self.language_hidden_dim = language_hidden_dim
        self.scale_factor = scale_factor
        self.input_dim = vision_hidden_dim * (scale_factor**2)
        self.output_dim = language_hidden_dim

        self.proj = nn.Linear(self.input_dim, self.output_dim, bias=False)

    def pixel_shuffle(self, x):
        """
        TODO:
        1. Reshape the sequence of vision tokens into a square grid.
        2. Group neighboring tokens according to `self.scale_factor`.
        3. Concatenate their channel dimensions.
        4. Return a sequence with fewer tokens and a larger hidden dimension.
        """
        raise NotImplementedError("Implement pixel_shuffle in PP7/models/modality_projector.py")

    def forward(self, x):
        """
        TODO:
        1. Apply `pixel_shuffle`.
        2. Project the shuffled features to the language hidden size.
        """
        raise NotImplementedError("Implement forward in PP7/models/modality_projector.py")

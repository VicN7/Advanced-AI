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
        batch_size, seq_len, hidden_dim = x.size()
        seq_root = int(seq_len**0.5)

        assert seq_root * seq_root == seq_len, "The number of image tokens must be a square."
        assert seq_root % self.scale_factor == 0, "The scale factor must divide the token grid."

        x = x.view(batch_size, seq_root, seq_root, hidden_dim)

        out_h = seq_root // self.scale_factor
        out_w = seq_root // self.scale_factor
        x = x.reshape(
            batch_size,
            out_h,
            self.scale_factor,
            out_w,
            self.scale_factor,
            hidden_dim,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(batch_size, out_h * out_w, hidden_dim * (self.scale_factor**2))
        return x

    def forward(self, x):
        x = self.pixel_shuffle(x)
        x = self.proj(x)
        return x

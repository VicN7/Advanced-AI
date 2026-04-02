import torch
import torch.nn as nn

from transformers import AutoModel, AutoModelForCausalLM

from data.processors import get_image_processor, get_tokenizer
from models.config import VLMConfig
from models.modality_projector import ModalityProjector
from models.utils import top_k_top_p_filtering


class VisionLanguageModel(nn.Module):
    def __init__(self, cfg: VLMConfig):
        super().__init__()
        self.cfg = cfg

        self.tokenizer = get_tokenizer(cfg.lm_tokenizer, cfg.image_token)
        self.image_processor = get_image_processor(cfg.vit_model_type)

        self.vision_backbone = AutoModel.from_pretrained(cfg.vit_model_type)
        if hasattr(self.vision_backbone, "vision_model"):
            self.vision_backbone = self.vision_backbone.vision_model

        self.language_model = AutoModelForCausalLM.from_pretrained(cfg.lm_model_type)
        self.language_model.resize_token_embeddings(len(self.tokenizer))

        vision_hidden_dim = self.vision_backbone.config.hidden_size
        language_hidden_dim = self.language_model.config.hidden_size
        self.modality_projector = ModalityProjector(
            vision_hidden_dim=vision_hidden_dim,
            language_hidden_dim=language_hidden_dim,
            scale_factor=cfg.mp_pixel_shuffle_factor,
        )

        image_size = self.vision_backbone.config.image_size
        patch_size = self.vision_backbone.config.patch_size
        grid_size = image_size // patch_size
        self.cfg.mp_image_token_length = (
            grid_size // self.cfg.mp_pixel_shuffle_factor
        ) ** 2

    def _replace_img_tokens_with_embd(self, input_ids, token_embd, image_embd):
        """
        TODO:
        Replace every `<|image|>` token embedding with the projected image embeddings.
        The final tensor must keep shape `[batch_size, seq_len, lm_hidden_dim]`.
        """
        raise NotImplementedError(
            "Implement _replace_img_tokens_with_embd in PP7/models/vision_language_model.py"
        )

    def forward(self, input_ids, pixel_values, attention_mask=None, labels=None):
        """
        TODO:
        1. Embed the input tokens with the language model input embedding matrix.
        2. Run the vision backbone on `pixel_values`.
        3. Project the vision features with the modality projector.
        4. Replace the image placeholder token embeddings.
        5. Call the language model with `inputs_embeds=...`.
        6. Return `(logits, loss)`.
        """
        raise NotImplementedError(
            "Implement forward in PP7/models/vision_language_model.py"
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids,
        pixel_values,
        attention_mask=None,
        max_new_tokens=30,
        top_k=50,
        top_p=0.9,
        temperature=0.8,
        greedy=False,
    ):
        """
        TODO:
        Implement autoregressive generation with:
        1. a multimodal prefill pass,
        2. repeated sampling of the next token,
        3. a KV cache passed through the Hugging Face language model.
        """
        raise NotImplementedError(
            "Implement generate in PP7/models/vision_language_model.py"
        )

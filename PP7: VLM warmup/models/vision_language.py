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
        )

        image_size = self.vision_backbone.config.image_size
        patch_size = self.vision_backbone.config.patch_size
        grid_size = image_size // patch_size
        self.cfg.mp_image_token_length = grid_size**2

    def _replace_img_tokens_with_embd(self, input_ids, token_embd, image_embd):
        updated_token_embd = token_embd.clone()
        image_mask = input_ids == self.tokenizer.image_token_id
        updated_token_embd[image_mask] = image_embd.reshape(-1, image_embd.size(-1)).to(
            updated_token_embd.dtype
        )
        return updated_token_embd

    def forward(self, input_ids, pixel_values, attention_mask=None, labels=None):
        token_embd = self.language_model.get_input_embeddings()(input_ids)

        vision_outputs = self.vision_backbone(
            pixel_values=pixel_values,
            return_dict=True,
        )
        image_embd = self.modality_projector(vision_outputs.last_hidden_state)
        token_embd = self._replace_img_tokens_with_embd(input_ids, token_embd, image_embd)

        outputs = self.language_model(
            inputs_embeds=token_embd,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=True,
            return_dict=True,
        )
        return outputs.logits, outputs.loss

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
        token_embd = self.language_model.get_input_embeddings()(input_ids)

        vision_outputs = self.vision_backbone(
            pixel_values=pixel_values,
            return_dict=True,
        )
        image_embd = self.modality_projector(vision_outputs.last_hidden_state)
        token_embd = self._replace_img_tokens_with_embd(input_ids, token_embd, image_embd)

        outputs = self.language_model(
            inputs_embeds=token_embd,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        generated = []
        current_attention_mask = attention_mask

        for _ in range(max_new_tokens):
            if greedy:
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                filtered_logits = top_k_top_p_filtering(
                    logits / temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                probs = torch.softmax(filtered_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)

            generated.append(next_token_id)

            next_token_embd = self.language_model.get_input_embeddings()(next_token_id)
            if current_attention_mask is not None:
                current_attention_mask = torch.cat(
                    [
                        current_attention_mask,
                        torch.ones(
                            (current_attention_mask.size(0), 1),
                            dtype=current_attention_mask.dtype,
                            device=current_attention_mask.device,
                        ),
                    ],
                    dim=1,
                )

            outputs = self.language_model(
                inputs_embeds=next_token_embd,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

        if not generated:
            return torch.empty((input_ids.size(0), 0), dtype=torch.long, device=input_ids.device)
        return torch.cat(generated, dim=1)

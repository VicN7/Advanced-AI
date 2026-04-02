import logging

import torch
from PIL import Image
from torch.utils.data import Dataset

from data.processors import get_image_string


class VQADataset(Dataset):
    def __init__(self, dataset, tokenizer, image_processor, mp_image_token_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mp_image_token_length = mp_image_token_length
        self.caption_prompt = "Describe the image."

    def __len__(self):
        return len(self.dataset)

    def _process_image(self, image):
        if not isinstance(image, Image.Image):
            raise ValueError(f"Expected a PIL image, got {type(image)}")

        if image.mode != "RGB":
            image = image.convert("RGB")

        pixel_values = self.image_processor(images=image, return_tensors="pt")[
            "pixel_values"
        ]
        return pixel_values.squeeze(0)

    def _prepare_inputs_and_loss_mask(self, messages):
        conv_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_special_tokens=False,
            return_dict=True,
        )
        mask = [0] * len(conv_ids["input_ids"])

        for idx, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue

            prefix_ids = self.tokenizer.apply_chat_template(
                messages[:idx],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
            )["input_ids"]
            full_ids = self.tokenizer.apply_chat_template(
                messages[: idx + 1],
                tokenize=True,
                add_special_tokens=False,
                return_dict=True,
            )["input_ids"]

            start = len(prefix_ids)
            end = len(full_ids)
            if end > start:
                mask[start:end] = [1] * (end - start)

        input_ids = torch.tensor(conv_ids["input_ids"])
        attention_mask = torch.tensor(conv_ids["attention_mask"])
        loss_mask = torch.tensor(mask, dtype=torch.bool)
        return input_ids, attention_mask, loss_mask

    def _build_cauldron_messages(self, item):
        messages = []
        for text in item["texts"]:
            messages.append({"role": "user", "content": text["user"]})
            messages.append({"role": "assistant", "content": text["assistant"]})

        if not messages:
            return []
        return messages

    def _build_flickr_messages(self, item):
        captions = item.get("original_alt_text") or item.get("alt_text") or []
        if isinstance(captions, str):
            captions = [captions]
        if not captions:
            return []

        caption = next((caption.strip() for caption in captions if caption and caption.strip()), None)
        if caption is None:
            return []

        return [
            {"role": "user", "content": self.caption_prompt},
            {"role": "assistant", "content": caption},
        ]

    def _build_messages(self, item, has_image):
        if "texts" in item:
            messages = self._build_cauldron_messages(item)
        else:
            messages = self._build_flickr_messages(item)

        if not messages:
            return []

        for msg in messages:
            if self.tokenizer.image_token in msg["content"]:
                logging.warning("Removed an unexpected image token from the text.")
                msg["content"] = msg["content"].replace(self.tokenizer.image_token, "")

        if has_image:
            image_string = get_image_string(self.tokenizer, self.mp_image_token_length)
            messages[0]["content"] = image_string + messages[0]["content"]

        return messages

    def __getitem__(self, idx):
        item = self.dataset[idx]

        images = item.get("images")
        if images is None and "image" in item:
            images = item["image"]
        if images is None:
            return None
        if not isinstance(images, list):
            images = [images]
        if len(images) == 0:
            return None

        pixel_values = self._process_image(images[0])
        messages = self._build_messages(item, has_image=True)
        if not messages:
            return None

        input_ids, attention_mask, loss_mask = self._prepare_inputs_and_loss_mask(
            messages
        )
        labels = input_ids.clone().masked_fill(~loss_mask, -100)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

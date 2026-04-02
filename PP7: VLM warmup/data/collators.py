import torch


class VQACollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        batch = [sample for sample in batch if sample is not None]
        if not batch:
            return {
                "input_ids": torch.empty(0, dtype=torch.long),
                "attention_mask": torch.empty(0, dtype=torch.long),
                "labels": torch.empty(0, dtype=torch.long),
                "pixel_values": torch.empty(0),
            }

        batch = [sample for sample in batch if len(sample["input_ids"]) <= self.max_length]
        if not batch:
            return {
                "input_ids": torch.empty(0, dtype=torch.long),
                "attention_mask": torch.empty(0, dtype=torch.long),
                "labels": torch.empty(0, dtype=torch.long),
                "pixel_values": torch.empty(0),
            }

        max_len = max(len(sample["input_ids"]) for sample in batch)

        input_ids = []
        attention_masks = []
        labels = []
        pixel_values = []

        for sample in batch:
            pad_len = max_len - len(sample["input_ids"])
            input_ids.append(
                torch.nn.functional.pad(
                    sample["input_ids"],
                    (0, pad_len),
                    value=self.tokenizer.pad_token_id,
                )
            )
            attention_masks.append(
                torch.nn.functional.pad(
                    sample["attention_mask"],
                    (0, pad_len),
                    value=0,
                )
            )
            labels.append(
                torch.nn.functional.pad(
                    sample["labels"],
                    (0, pad_len),
                    value=-100,
                )
            )
            pixel_values.append(sample["pixel_values"])

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(labels),
            "pixel_values": torch.stack(pixel_values),
        }

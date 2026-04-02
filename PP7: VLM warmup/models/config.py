from dataclasses import dataclass


@dataclass
class VLMConfig:
    vit_model_type: str = "google/siglip2-base-patch16-512"
    lm_model_type: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    lm_tokenizer: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    image_token: str = "<|image|>"
    mp_image_token_length: int = 64
    lm_max_length: int = 1024


@dataclass
class TrainConfig:
    dataset_path: str = "AnyModal/flickr30k"
    dataset_names: tuple[str, ...] = ()
    train_samples: int = 256
    val_samples: int = 64
    batch_size: int = 2
    max_steps: int = 20
    eval_interval: int = 10
    gradient_accumulation_steps: int = 1
    lr_projector: float = 1e-3
    lr_vision: float = 0.0
    lr_language: float = 0.0
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    num_workers: int = 0
    compile: bool = False
    split_seed: int = 0
    output_dir: str = "checkpoints"
    output_name: str = "projector.pt"

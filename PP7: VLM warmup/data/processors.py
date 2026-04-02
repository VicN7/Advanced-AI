from transformers import AutoImageProcessor, AutoTokenizer


TOKENIZER_CACHE = {}
IMAGE_PROCESSOR_CACHE = {}


def get_tokenizer(model_name, image_token):
    if model_name not in TOKENIZER_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        vocab = tokenizer.get_vocab()
        if image_token not in vocab:
            tokenizer.add_special_tokens(
                {"additional_special_tokens": [image_token]}
            )

        tokenizer.image_token = image_token
        tokenizer.image_token_id = tokenizer.convert_tokens_to_ids(image_token)
        TOKENIZER_CACHE[model_name] = tokenizer

    return TOKENIZER_CACHE[model_name]


def get_image_processor(model_name):
    if model_name not in IMAGE_PROCESSOR_CACHE:
        IMAGE_PROCESSOR_CACHE[model_name] = AutoImageProcessor.from_pretrained(
            model_name
        )
    return IMAGE_PROCESSOR_CACHE[model_name]


def get_image_string(tokenizer, num_image_tokens):
    return tokenizer.image_token * num_image_tokens

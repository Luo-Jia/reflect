import logging

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen3-8B"
logger = logging.getLogger(__name__)

def load_model():

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    dtype = torch.float16 if device == "mps" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    model.to(device)
    model.eval()

    return model, tokenizer, device


def generate_response(
    model,
    tokenizer,
    device,
    messages,
    max_new_tokens=512,
):

    # Tokenize conversation
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False,
        return_dict=True,
    )

    # Extract input IDs from BatchEncoding
    input_ids = inputs["input_ids"]

    # Validate tensor
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError(
            f"Expected torch.Tensor, got {type(input_ids)}"
        )

    if input_ids.ndim != 2:
        raise ValueError(
            f"Expected 2D tensor, got {input_ids.shape}"
        )

    if input_ids.shape[1] == 0:
        raise ValueError("Input tensor is empty.")

    # Move inputs to MPS
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
        if isinstance(value, torch.Tensor)
    }

    input_ids = inputs["input_ids"]

    logger.debug("Input shape: %s", input_ids.shape)
    logger.debug("Input device: %s", input_ids.device)

    # Generate response
    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Extract generated tokens only
    generated_tokens = outputs[
        0, input_ids.shape[1]:
    ]
    logger.debug("Generated tokens: %s; limit: %s; limit reached: %s",
                 generated_tokens.shape[0], max_new_tokens,
                 generated_tokens.shape[0] >= max_new_tokens)

    # Decode response
    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return response.strip()

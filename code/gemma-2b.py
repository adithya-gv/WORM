from gemma_archive.gemma.config import GemmaConfig, get_config_for_2b
from gemma_archive.gemma.model import GemmaModel, GemmaForCausalLM
from gemma_archive.gemma.tokenizer import Tokenizer

import sys
import os 
import contextlib
import torch

VARIANT = "2b"

@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)
    
model_config = get_config_for_2b()
model_config.tokenizer = "code/gemma_archive/tokenizer.model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with _set_default_tensor_type(torch.bfloat16):
    model = GemmaForCausalLM(model_config)
    checkpoint_path = "code/gemma_archive/gemma-2b.ckpt"
    model.load_weights(checkpoint_path)
    model = model.to(device).eval()

USER_CHAT_TEMPLATE = "<start_of_turn>user\n{prompt}<end_of_turn>\n"
MODEL_CHAT_TEMPLATE = "<start_of_turn>model\n{prompt}<end_of_turn>\n"

prompt = (
    USER_CHAT_TEMPLATE.format(
        prompt="What is a good place for travel in the US?"
    )
    + MODEL_CHAT_TEMPLATE.format(prompt="California.")
    + USER_CHAT_TEMPLATE.format(prompt="What can I do in California?")
    + "<start_of_turn>model\n"
)

output = model.generate(
    USER_CHAT_TEMPLATE.format(prompt=prompt),
    device=device,
    output_len=100,
)

print(output)

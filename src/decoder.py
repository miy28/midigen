# holds the gpt2 model itself

from transformers import GPT2LMHeadModel, GPT2Config
import torch
import torch.nn

# hugging face gpt2 model with empty weights

def melody_transformer():

    config = GPT2Config(
        vocab_size = 5000,
        n_positions = 512, # max sequence length (number of tokens)
        n_ctx = 512,
        n_layer = 6,
        n_head = 6,
        n_embd = 384
    )
    
    model = GPT2LMHeadModel(config) # load naked GPT2 model

    return model



# gpt2-small

def melody_transformer_pretrained_small():
    
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    return model
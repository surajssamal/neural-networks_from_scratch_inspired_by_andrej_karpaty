#!/bin/python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

from dataclasses import dataclass
from transformers import GPT2LMHeadModel


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd,3* config.n_embd)
        self.c_proj = nn.Linear(config.n_embd,  config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        #not really a bias more of a mask ,but following the OpenAI implementation
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                              .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        #compute attention scores
        scores = q@ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
        #apply attention mask
        scores = scores.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        #apply softmax
        attn = F.softmax(scores.float(), dim=-1).type(scores.type())
        #compute attention output
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(out)
        return out

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd,config.n_embd)
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class Gpt2_config:
    vocab_size: int = 50257
    block_size: int = 1024
    n_embd: int  = 768
    n_head: int = 12
    n_layer: int = 12
    dropout: float = 0.2

class Gpt2(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h =nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    def forward(self, input_ids):
        b, t = input_ids.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=input_ids.device)
        tok_emb = self.transformer.wte(input_ids)
        pos_emb =  self.transformer.wpe(pos)

        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits
    @classmethod
    def from_pretrained(cls,model_type):
        #load pretrained GPT-2 model weights from huggingface
        assert model_type in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
        from transformers import GPT2LMHeadModel
        print("loading weights from the pretrained gpt : %s"%model_type)

        #n_layer ,n_head ,n_embd determined from the model_type
        config_args = {
            "gpt2" : dict(n_layer=12,n_head=12,n_embd=768),#124m parameters
            "gpt2-medium" : dict(n_layer=24,n_head=16,n_embd=1024),#350m parameters
            "gpt2-xl" : dict(n_layer=36,n_head=16,n_embd=1280)#760m parameters
        
        }[model_type]
        config_args["vocab_size"] = 50257 #always 50257 
        config_args["block_size"] = 1024 # always 1024
        config  = Gpt2_config(**config_args)
        model = Gpt2(config)
        sd = model.state_dict()
        sd_keys =sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.masked_bias")] 
        #init a hugging face transformer model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")] #just removing the masking like we did up
        sd_keys_hf = [k for k in sd_hf if not k.endswith(".attn.bias")]
        transposed = ["attn.c_attn.weight","attn.c_proj.weight","mlp.c_fc.weight", "mlp.c_proj.weight"]
        


        # # basicallt the openai checkpoint is a prefix of the hugging face checkpoint
        # assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
                    
   
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device {device}")

num_return_sequences =5 
max_length = 30
model = Gpt2.from_pretrained("gpt2")
model.eval()
model.to(device)

#prefix tokens
import tiktoken
enc = tiktoken.get_encoding("gpt2")
tokens =enc.encode("my name is ")
tokens = torch.tensor(tokens,dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences,1)
x =tokens.to(device)
while x.size(1) < max_length:
    #forward pass
    with torch.no_grad():
        logits = model(x)
        #sample from the distribution
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)
        ix = torch.multinomial(topk_probs, num_samples=1)
        xcol =torch.gather(topk_indices, -1, ix)
        x = torch.cat([x, xcol], dim=1)
for i in range(num_return_sequences):
    tokens = x[i,:max_length].tolist()
    decoded = enc.decode(tokens)
    print(decoded)

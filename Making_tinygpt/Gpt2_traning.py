#!/bin/python3

import torch
import torch.nn as nn
import inspect 
import torch.nn.functional as F
import math

import tiktoken
from dataclasses import dataclass


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd,3* config.n_embd)
        self.c_proj = nn.Linear(config.n_embd,  config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT =1
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

        # scores = q@ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
        # #apply attention mask
        # scores = scores.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # #apply softmax
        # attn = F.softmax(scores.float(), dim=-1).type(scores.type())
        # #compute attention output
        # out = attn @ v
        # the reason i commented all this is cause which im going to do is use a flash attenntion which you can understand by learning kernal fusion and flash attention paper 
        # https://arxiv.org/pdf/2205.14135.pdf
        # https://github.com/HazyResearch/flash-attention
        # https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attention.py
        out = F.scaled_dot_product_attention(q,k,v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(out)
        return out

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd,config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT =1
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
        x = x + self.attn(self.ln_1(x)) #this is Resedual connection aka RESNET
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class Gpt2_config:
    vocab_size: int = 50257
    block_size: int =32 
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
        # weigth sharing scheme of the lm_head and wte
        self.transformer.wte.weight = self.lm_head.weight

        #now making the initaliazation 0.2 
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std =0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2*self.config.n_layer)**-0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            

    def forward(self, input_ids,target_ids=None):
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
        loss =None
        if target_ids is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        return logits, loss
    def config_optim(self,weight_decay,device,learning_rate):
        param_dict = {pn:p for pn,p in self.named_parameters() }
        param_dict = {pn:p for pn,p in param_dict.items() if p.requires_grad}
        #making sure if all the matmul parameters weight get decay not the baise and the embedding weight
        decay_weight =[p for n,p in param_dict.items() if p.dim() >= 2]
        decay_bias = [p for n,p in param_dict.items() if p.dim() <2]
        optim_group =[
            {"params":decay_weight,"weight_decay":weight_decay},
            {"params":decay_bias,"weight_decay":0.0}
        ] 
        num_decay_param = sum([p.numel() for p in decay_weight])
        num_bias_param = sum([p.numel() for p in decay_bias])
        print(f"num decayed {len(decay_weight)} params {num_decay_param} | num bias {len(decay_bias)} params {num_bias_param}")
        
        #Create adamw optimizer and use the fused version if it is availabe
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused= fused_available and device.type == "cuda"
        print(f"using fused adamw {use_fused}")
        optimizer = torch.optim.AdamW(optim_group,lr=learning_rate,betas=(0.9,0.95),eps=1e-6,weight_decay=weight_decay,fused=use_fused)
        return optimizer
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
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
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

# num_return_sequences =5 
# max_length = 30
# model = Gpt2.from_pretrained("gpt2")
# model.eval()
# model.to(device)
#
#prefix tokens
#Making the batch function
# n1 = len(encoded_data)*0.8
# n2 = len(encoded_data)*0.9

# traning_data = encoded_data[:int(n1)]
# test_data  = encoded_data[int(n1):int(n2)]
# val_data  = encoded_data[int(n2):]
# print(len(traning_data),len(test_data),len(val_data))

#creating a datraloader
class Dataloader:
    def __init__(self,batch_size,seq_len):
        self.B=batch_size
        self.T=seq_len
        
        with open("input.txt","r") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        encoded_data = enc.encode(text)
        self.data = torch.tensor(encoded_data)
        print(f"len of tokens {len(self.data)}")
        print(f"1 epoc {len(self.data)//self.B*self.T}")
        
        self.current_position = 0
    def next_batch(self):
        B,T = self.B,self.T
        buf = self.data[self.current_position : self.current_position + B*T+1]
        x = buf[:-1].view(B,T)
        y = buf[1:].view(B,T)

        self.current_position += B*T
        if self.current_position >= len(self.data):
            self.current_position = 0
        return x,y

model = Gpt2(Gpt2_config())
model.to(device)
model= torch.compile(model)
max_lr =3e-4
min_lr = max_lr*0.1
warm_up_steps = 10
max_steps =50
#making a learing rate decay 
def get_lr(step):
    if step < warm_up_steps:
        return max_lr*(step+1)/warm_up_steps
    if steps>max_steps:
        return min_lr
    decay_ratio = (step-warm_up_steps)/(max_steps-warm_up_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
optimizer = model.config_optim(weight_decay=0.1,device=device,learning_rate=max_lr) 
import time
# now what im doing is make a batch accumilation method for better optimization for that were goona use batch
Batch_size =2560
B=16
T=8
assert Batch_size % (B*T) == 0
grad_accum_steps = Batch_size // (B*T)
print(f"grad_accum_steps {grad_accum_steps}| total batch_size {Batch_size}")



train_data =Dataloader(batch_size=B,seq_len=T)
torch.set_float32_matmul_precision("high") #this is for using the fp32 in nvidia cuda so that it doesnt take much of the GPU memories 
for steps in range(max_steps):

    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0
    for _ in range(grad_accum_steps):
        x,y = train_data.next_batch()
        x,y =x.to(device),y.to(device)
        logits,loss = model(x,y)
        loss = loss / grad_accum_steps
        loss_accum +=loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(steps)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1-t0)*1000
    tokens_per_sec = (train_data.B*train_data.T)/(t1-t0)
    print(f"epoc {steps} loss {loss_accum:.9f} | norm {torch.norm(logits).item():.4f} |dt {dt:.2f} | tokens per sec {tokens_per_sec:.2f}")

import sys
sys.exit()

tokens = enc.encode("my name is ")
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

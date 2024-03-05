import torch
import torch.nn as nn
from torch.nn import functional as F

from model import Block
from utils import *

seed_everything(42)

# hyperparameters
batch_size = 8 
block_size = 32 
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
embedding_size = 32
num_heads = 2
n_layer = 2
dropout = 0.1

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# get all the unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mappings of characters to integers and vice versa..
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # string to int conversion
decode = lambda l: ''.join([itos[i] for i in l]) # into to string conversion

# create train,test splits (90%/10%)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


class GPT(nn.Module):

    def __init__(self, vocab_size, embedding_size, block_size, num_heads, n_layer):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.positional_embeddings = nn.Embedding(block_size, embedding_size)
        self.blocks = nn.Sequential(*[Block(embedding_size, num_heads, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(embedding_size) # final layer norm
        self.lm_head = nn.Linear(embedding_size, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets (tesnor) with shape (B,T) 
        toks = self.embeddings(idx) # (B, T, E)
        pos = self.positional_embeddings(torch.arange(T, device=device)) # [T, E]
        x = toks + pos           # [B, T, E]
        x = self.blocks(x)       # [B, T, E]
        x = self.ln_f(x)         # [B, T, E]
        logits = self.lm_head(x) # [B, T, vocab_size]

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPT(
    vocab_size=vocab_size,
    embedding_size=embedding_size,
    block_size=block_size,
    num_heads=num_heads,
    n_layer=n_layer
)
model = model.to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'Million parameters')

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # get losses
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model, eval_iters, train_data, val_data, batch_size, block_size, device)
        print(f"step [{iter}/{max_iters}]: train loss {losses['train']:.5f} | valid loss {losses['val']:.5f}")

    # sample the batch of data
    xb, yb = get_batch('train', train_data, val_data, batch_size, block_size, device)

    # backprop
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# inference
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

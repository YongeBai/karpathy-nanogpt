import torch
import torch.nn as nn
from torch.nn import functional as F

from typing import Union, Tuple, Optional

batch_size = 64
context_size = 256
num_epochs = 5000
eval_interval = 500
eval_epochs = 200
n_embed = 384
n_heads = 6
n_layer = 6
dropout_p = 0.2
lr = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

with open('./input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[t] for t in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split: str):
    data = train_data if split=='train' else val_data
    indexes = torch.randint(0, len(data) - context_size, (batch_size,))
    x = torch.stack([data[i:i+context_size] for i in indexes])
    y = torch.stack([data[i+1:i+context_size+1] for i in indexes])
    x, y = x.to(device), y.to(device)
    return x, y

class Head(nn.Module):

    def __init__(self, head_size: int):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dropout_p)
        self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k, q = self.key(x), self.query(x)
        weights = q @ k.transpose(-2, -1) * C**-0.5
        weights = weights.masked_fill(self.tril[:T, :T]==0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        v = self.value(x)
        output = weights @ v
        return output

class MultiHeadAttention(nn.Module):

    def __init__(self, num_head: int, head_size: int):
        super().__init__()
        self.proj = nn.Linear(n_embed, n_embed)
        self.heads = nn.ModuleList(
            [Head(head_size=head_size) for _ in range(num_head)]
        )
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output =  torch.cat([h(x) for h in self.heads], dim=-1)
        output = self.dropout(self.proj(output))
        return output

class FeedForward(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed)
        )
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.dropout(self.net(x))
        return output

class TransformerBlock(nn.Module):

    def __init__(self, num_heads):
        super().__init__()
        head_size = n_embed//num_heads
        self.sa_head = MultiHeadAttention(num_head=num_heads, head_size=head_size)
        self.ff = FeedForward()
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
    

class GPT(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(context_size, n_embed)
        self.blocks = nn.Sequential(
            *[TransformerBlock(num_heads=n_heads) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(
            self,
            idx: torch.Tensor, 
            targets: Optional[torch.Tensor] = None
            ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        token_embeddings = self.token_embedding_table(idx)
        position_embeddings = self.position_embedding_table(torch.arange(T, device=device))        
        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(
            self,
            idx: torch.Tensor,
            max_new_tokens: int
            ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probabilities = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx            
                        
@torch.no_grad()
def estimate_loss():
    output = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_epochs)
        for k in range(eval_epochs):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        output[split] = losses.mean()
    model.train()
    return output

model = GPT()
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for epoch in range(num_epochs+1):
    if epoch % eval_interval == 0:
        mean_losses = estimate_loss()
        print(f"Epoch {epoch}: train loss={mean_losses['train']:.4f} val loss={mean_losses['val']:.4f}")
    optimizer.zero_grad()
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

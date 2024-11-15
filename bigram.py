import torch
import torch.nn as nn
from torch.nn import functional as F

from typing import Union, Tuple, Optional

batch_size = 32
context_size = 8
num_epochs = 3000
eval_interval = 300
eval_epochs = 200
lr = 1e-2
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

class BigramLM(nn.Module):
    
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(
            self,
            idx: torch.Tensor, 
            targets: Optional[torch.Tensor] = None
            ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        logits = self.token_embedding_table(idx)

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
            logits, loss = self(idx)
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

model = BigramLM(vocab_size=vocab_size)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for epoch in range(num_epochs):
    if epoch % eval_interval == 0:
        mean_losses = estimate_loss()
        print(f"Epoch {epoch}: train loss={mean_losses['train']:.4f} val loss={mean_losses['val']:.4f}")
    optimizer.zero_grad()
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(model.generate(context, max_new_tokens=500).shape)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

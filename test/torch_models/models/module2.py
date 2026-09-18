import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np

# ─────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────
class Config:
    vocab_size    = 10000
    embed_dim     = 256
    num_heads     = 8
    num_layers    = 6
    ff_dim        = 1024
    max_seq_len   = 512
    dropout       = 0.1
    batch_size    = 16
    epochs        = 20
    lr            = 3e-4
    device        = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()

# ─────────────────────────────────────────
# 2. TOKENIZER (character or word level)
# ─────────────────────────────────────────
class SimpleTokenizer:
    def __init__(self, text, vocab_size=10000):
        words = text.lower().split()
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        # most common words
        vocab = sorted(freq, key=freq.get, reverse=True)[:vocab_size - 4]#type:ignore
        self.word2idx = {"<PAD>":0, "<UNK>":1, "<BOS>":2, "<EOS>":3}
        for i, w in enumerate(vocab):
            self.word2idx[w] = i + 4
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

    def encode(self, text):
        return [self.word2idx.get(w, 1) for w in text.lower().split()]

    def decode(self, ids):
        return " ".join(self.idx2word.get(i, "<UNK>") for i in ids)

# ─────────────────────────────────────────
# 3. DATASET
# ─────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, token_ids, seq_len):
        self.data = token_ids
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:],  dtype=torch.long)
        return x, y

# ─────────────────────────────────────────
# 4. ATTENTION
# ─────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.h    = num_heads
        self.d_k  = embed_dim // num_heads
        self.Wq   = nn.Linear(embed_dim, embed_dim)
        self.Wk   = nn.Linear(embed_dim, embed_dim)
        self.Wv   = nn.Linear(embed_dim, embed_dim)
        self.Wo   = nn.Linear(embed_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        # split into heads
        Q = self.Wq(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        K = self.Wk(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        V = self.Wv(x).view(B, T, self.h, self.d_k).transpose(1, 2)

        # scaled dot-product attention
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = self.drop(F.softmax(scores, dim=-1))

        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, C)
        return self.Wo(out)

# ─────────────────────────────────────────
# 5. DECODER BLOCK
# ─────────────────────────────────────────
class DecoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn  = MultiHeadAttention(cfg.embed_dim, cfg.num_heads, cfg.dropout)
        self.ff    = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.ff_dim),
            nn.GELU(),
            nn.Linear(cfg.ff_dim, cfg.embed_dim),
            nn.Dropout(cfg.dropout)
        )
        self.norm1 = nn.LayerNorm(cfg.embed_dim)
        self.norm2 = nn.LayerNorm(cfg.embed_dim)
        self.drop  = nn.Dropout(cfg.dropout)

    def forward(self, x, mask=None):
        # self-attention + residual
        x = x + self.drop(self.attn(self.norm1(x), mask))
        # feedforward + residual
        x = x + self.ff(self.norm2(x))
        return x
# ─────────────────────────────────────────
# 6. FULL DECODER MODEL
# ─────────────────────────────────────────
class KidnetGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb  = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_emb  = nn.Embedding(cfg.max_seq_len, cfg.embed_dim)
        self.drop     = nn.Dropout(cfg.dropout)
        self.blocks   = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm     = nn.LayerNorm(cfg.embed_dim)
        self.head     = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)

        # weight tying (token embedding & output head share weights)
        self.head.weight = self.tok_emb.weight

    def forward(self, x, targets=None):
        B, T = x.shape
        pos  = torch.arange(T, device=x.device).unsqueeze(0)

        # causal mask — decoder can only see past tokens
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        out  = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        for block in self.blocks:
            out = block(out, mask)
        out  = self.norm(out)
        logits = self.head(out)                          # (B, T, vocab)

        loss = None 
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0                           # ignore PAD
            )
        return logits, loss

# ─────────────────────────────────────────
# 7. GENERATION
# ─────────────────────────────────────────
@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8, top_k=50):
    model.eval()
    ids = [2] + tokenizer.encode(prompt)                # prepend <BOS>
    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(cfg.device)

    for _ in range(max_new_tokens):
        # crop if too long
        inp = ids[:, -cfg.max_seq_len:]
        logits, _ = model(inp)
        logits = logits[:, -1, :] / temperature         # last token logits

        # top-k filtering
        if top_k:
            vals, _ = torch.topk(logits, top_k)
            logits[logits < vals[:, -1:]] = float('-inf')

        probs    = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        ids      = torch.cat([ids, next_tok], dim=1)

        if next_tok.item() == 3:                        # <EOS>
            break

    return tokenizer.decode(ids[0].tolist()[1:])        # skip <BOS>

# ─────────────────────────────────────────
# 8. TRAINING LOOP
# ─────────────────────────────────────────
def train(model, loader, optimizer, scheduler, epoch):
    model.train()
    total_loss = 0
    for step, (x, y) in enumerate(loader):
        x, y = x.to(cfg.device), y.to(cfg.device)
        _, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        if step % 50 == 0:
            print(f"Epoch {epoch} | Step {step} | Loss {loss.item():.4f}")

    return total_loss / len(loader)

# ─────────────────────────────────────────
# 9. MAIN — plug in your kidnet.txt
# ─────────────────────────────────────────
if __name__ == "__main__":
    # Load your data
    with open("kidney.txt", "r") as f:
        text = f.read()

    tokenizer  = SimpleTokenizer(text, cfg.vocab_size)
    cfg.vocab_size = tokenizer.vocab_size          # update to actual vocab size
    token_ids  = tokenizer.encode(text)

    dataset    = TextDataset(token_ids, cfg.max_seq_len)
    loader     = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model      = KidnetGPT(cfg).to(cfg.device)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=cfg.epochs * len(loader))

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training on: {cfg.device}")

    for epoch in range(1, cfg.epochs + 1):
        avg_loss = train(model, loader, optimizer, scheduler, epoch)
        print(f"── Epoch {epoch} avg loss: {avg_loss:.4f}")

        # sample every 5 epochs
        if epoch % 5 == 0:
            out = generate(model, tokenizer, "the kidney", max_new_tokens=200)
            print(f"\nSample: {out}\n")

    torch.save(model.state_dict(), "kidnet_gpt.pt")
    print("Model saved.")

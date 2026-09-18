"""
KidnetGPT — JAX + Flax + Optax implementation.
Decoder-only, GPT-style causal language model with weight-tied output head.
"""

import math
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import train_state

# ─────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────
class Config:
    vocab_size  = 10000
    embed_dim   = 256
    num_heads   = 8
    num_layers  = 6
    ff_dim      = 1024
    max_seq_len = 512
    dropout     = 0.1
    batch_size  = 16
    epochs      = 20
    lr          = 3e-4

cfg = Config()

# ─────────────────────────────────────────
# 2. TOKENIZER (word level, framework-agnostic)
# ─────────────────────────────────────────
class SimpleTokenizer:
    def __init__(self, text, vocab_size=10000):
        words = text.lower().split()
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        vocab = sorted(freq, key=freq.get, reverse=True)[:vocab_size - 4]
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        for i, w in enumerate(vocab):
            self.word2idx[w] = i + 4
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

    def encode(self, text):
        return [self.word2idx.get(w, 1) for w in text.lower().split()]

    def decode(self, ids):
        return " ".join(self.idx2word.get(i, "<UNK>") for i in ids)

# ─────────────────────────────────────────
# 3. DATASET (numpy windows, batched manually)
# ─────────────────────────────────────────
def make_windows(token_ids, seq_len):
    data = np.array(token_ids, dtype=np.int32)
    n = len(data) - seq_len
    xs = np.stack([data[i:i + seq_len] for i in range(n)])
    ys = np.stack([data[i + 1:i + seq_len + 1] for i in range(n)])
    return xs, ys


def batch_iter(rng, xs, ys, batch_size):
    perm = np.array(jax.random.permutation(rng, xs.shape[0]))
    for i in range(0, len(perm) - batch_size + 1, batch_size):
        idx = perm[i:i + batch_size]
        yield jnp.array(xs[idx]), jnp.array(ys[idx])

# ─────────────────────────────────────────
# 4. ATTENTION
# ─────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    embed_dim: int
    num_heads: int
    dropout: float

    def setup(self):
        assert self.embed_dim % self.num_heads == 0
        self.h = self.num_heads
        self.d_k = self.embed_dim // self.num_heads
        self.Wq = nn.Dense(self.embed_dim)
        self.Wk = nn.Dense(self.embed_dim)
        self.Wv = nn.Dense(self.embed_dim)
        self.Wo = nn.Dense(self.embed_dim)
        self.drop = nn.Dropout(self.dropout)

    def __call__(self, x, mask, train: bool):
        B, T, C = x.shape
        Q = self.Wq(x).reshape(B, T, self.h, self.d_k).transpose(0, 2, 1, 3)
        K = self.Wk(x).reshape(B, T, self.h, self.d_k).transpose(0, 2, 1, 3)
        V = self.Wv(x).reshape(B, T, self.h, self.d_k).transpose(0, 2, 1, 3)

        scores = jnp.matmul(Q, jnp.swapaxes(K, -2, -1)) / math.sqrt(self.d_k)
        scores = jnp.where(mask == 0, -jnp.inf, scores)
        attn = jax.nn.softmax(scores, axis=-1)
        attn = self.drop(attn, deterministic=not train)

        out = jnp.matmul(attn, V).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.Wo(out)

# ─────────────────────────────────────────
# 5. DECODER BLOCK
# ─────────────────────────────────────────
class DecoderBlock(nn.Module):
    embed_dim: int
    num_heads: int
    ff_dim: int
    dropout: float

    def setup(self):
        self.attn = MultiHeadAttention(self.embed_dim, self.num_heads, self.dropout)
        self.ff1 = nn.Dense(self.ff_dim)
        self.ff2 = nn.Dense(self.embed_dim)
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.drop = nn.Dropout(self.dropout)

    def __call__(self, x, mask, train: bool):
        x = x + self.drop(self.attn(self.norm1(x), mask, train), deterministic=not train)
        ff_out = self.ff2(nn.gelu(self.ff1(self.norm2(x))))
        ff_out = self.drop(ff_out, deterministic=not train)
        x = x + ff_out
        return x

# ─────────────────────────────────────────
# 6. FULL DECODER MODEL (weight-tied head)
# ─────────────────────────────────────────
class KidnetGPT(nn.Module):
    vocab_size: int
    embed_dim: int
    num_heads: int
    num_layers: int
    ff_dim: int
    max_seq_len: int
    dropout: float

    def setup(self):
        self.tok_emb = nn.Embed(self.vocab_size, self.embed_dim)
        self.pos_emb = nn.Embed(self.max_seq_len, self.embed_dim)
        self.drop = nn.Dropout(self.dropout)
        self.blocks = [
            DecoderBlock(self.embed_dim, self.num_heads, self.ff_dim, self.dropout)
            for _ in range(self.num_layers)
        ]
        self.norm = nn.LayerNorm()

    def __call__(self, x, train: bool = True):
        B, T = x.shape
        pos = jnp.arange(T)[None, :]
        mask = jnp.tril(jnp.ones((T, T)))[None, None, :, :]

        out = self.drop(self.tok_emb(x) + self.pos_emb(pos), deterministic=not train)
        for block in self.blocks:
            out = block(out, mask, train)
        out = self.norm(out)
        logits = self.tok_emb.attend(out)  # weight-tied output head
        return logits

# ─────────────────────────────────────────
# 7. TRAIN STATE / LOSS / STEP
# ─────────────────────────────────────────
def make_state(rng, cfg):
    model = KidnetGPT(
        vocab_size=cfg.vocab_size, embed_dim=cfg.embed_dim, num_heads=cfg.num_heads,
        num_layers=cfg.num_layers, ff_dim=cfg.ff_dim, max_seq_len=cfg.max_seq_len,
        dropout=cfg.dropout,
    )
    dummy = jnp.ones((cfg.batch_size, cfg.max_seq_len), dtype=jnp.int32)
    rng, init_rng, drop_rng = jax.random.split(rng, 3)
    params = model.init({"params": init_rng, "dropout": drop_rng}, dummy, train=True)["params"]

    total_steps = cfg.epochs * 1000  # placeholder; recomputed after loader length known
    schedule = optax.cosine_decay_schedule(cfg.lr, total_steps)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule))
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return model, state


def loss_fn(params, apply_fn, x, y, dropout_rng):
    logits = apply_fn({"params": params}, x, train=True, rngs={"dropout": dropout_rng})
    mask = (y != 0).astype(jnp.float32)
    one_hot = jax.nn.one_hot(y, logits.shape[-1])
    per_tok = optax.softmax_cross_entropy(logits, one_hot)
    return (per_tok * mask).sum() / jnp.maximum(mask.sum(), 1.0)


@jax.jit
def train_step(state, x, y, dropout_rng):
    dropout_rng, step_rng = jax.random.split(dropout_rng)
    loss, grads = jax.value_and_grad(loss_fn)(state.params, state.apply_fn, x, y, step_rng)
    state = state.apply_gradients(grads=grads)
    return state, loss, dropout_rng

# ─────────────────────────────────────────
# 8. GENERATION (top-k sampling)
# ─────────────────────────────────────────
def generate(model, params, tokenizer, prompt, rng, max_new_tokens=200, temperature=0.8, top_k=50):
    ids = [2] + tokenizer.encode(prompt)  # prepend <BOS>
    ids = jnp.array(ids, dtype=jnp.int32)[None, :]

    for _ in range(max_new_tokens):
        inp = ids[:, -cfg.max_seq_len:]
        logits = model.apply({"params": params}, inp, train=False)
        logits = logits[:, -1, :] / temperature

        if top_k:
            top_vals, _ = jax.lax.top_k(logits, top_k)
            threshold = top_vals[:, -1:]
            logits = jnp.where(logits < threshold, -jnp.inf, logits)

        rng, sample_rng = jax.random.split(rng)
        next_tok = jax.random.categorical(sample_rng, logits, axis=-1)[:, None]
        ids = jnp.concatenate([ids, next_tok], axis=1)

        if int(next_tok[0, 0]) == 3:  # <EOS>
            break

    return tokenizer.decode(np.array(ids[0]).tolist()[1:])  # skip <BOS>

# ─────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    dataset_path = Path(__file__).resolve().parent.parent / "datasets" / "kidney.txt"
    with open(dataset_path, "r") as f:
        text = f.read()

    tokenizer = SimpleTokenizer(text, cfg.vocab_size)
    cfg.vocab_size = tokenizer.vocab_size
    token_ids = tokenizer.encode(text)

    xs, ys = make_windows(token_ids, cfg.max_seq_len)
    steps_per_epoch = max(1, (xs.shape[0] // cfg.batch_size))

    rng = jax.random.PRNGKey(0)
    model, state = make_state(rng, cfg)
    # rebuild schedule/tx now that steps_per_epoch is known
    schedule = optax.cosine_decay_schedule(cfg.lr, cfg.epochs * steps_per_epoch)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule))
    state = train_state.TrainState.create(apply_fn=state.apply_fn, params=state.params, tx=tx)

    n_params = sum(p.size for p in jax.tree_util.tree_leaves(state.params))
    print(f"Model parameters: {n_params:,}")
    print(f"Devices: {jax.devices()}")

    dropout_rng = jax.random.PRNGKey(1)
    for epoch in range(1, cfg.epochs + 1):
        rng, perm_rng = jax.random.split(rng)
        total_loss, n_steps = 0.0, 0
        for step, (x, y) in enumerate(batch_iter(perm_rng, xs, ys, cfg.batch_size)):
            state, loss, dropout_rng = train_step(state, x, y, dropout_rng)
            total_loss += float(loss)
            n_steps += 1
            if step % 50 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss {float(loss):.4f}")

        avg_loss = total_loss / max(n_steps, 1)
        print(f"── Epoch {epoch} avg loss: {avg_loss:.4f}")

        if epoch % 5 == 0:
            rng, gen_rng = jax.random.split(rng)
            out = generate(model, state.params, tokenizer, "the kidney", gen_rng, max_new_tokens=200)
            print(f"\nSample: {out}\n")

    np.save("kidnet_gpt_params.npy", jax.tree_util.tree_map(np.array, state.params), allow_pickle=True)
    print("Model saved.")

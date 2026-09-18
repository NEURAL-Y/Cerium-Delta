"""
KidnetGPT — TensorFlow + Keras implementation.
Decoder-only, GPT-style causal language model with weight-tied output head.
"""

import math
from pathlib import Path

import numpy as np
import tensorflow as tf

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

gpus = tf.config.list_physical_devices("GPU")
print(f"Using device: {'GPU' if gpus else 'CPU'}")

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
# 3. DATASET
# ─────────────────────────────────────────
def make_windows(token_ids, seq_len):
    data = np.array(token_ids, dtype=np.int32)
    n = len(data) - seq_len
    xs = np.stack([data[i:i + seq_len] for i in range(n)])
    ys = np.stack([data[i + 1:i + seq_len + 1] for i in range(n)])
    return xs, ys


def make_dataset(xs, ys, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((xs, ys))
    return ds.shuffle(buffer_size=len(xs)).batch(batch_size, drop_remainder=True)

# ─────────────────────────────────────────
# 4. ATTENTION
# ─────────────────────────────────────────
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.h = num_heads
        self.d_k = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.Wq = tf.keras.layers.Dense(embed_dim)
        self.Wk = tf.keras.layers.Dense(embed_dim)
        self.Wv = tf.keras.layers.Dense(embed_dim)
        self.Wo = tf.keras.layers.Dense(embed_dim)
        self.drop = tf.keras.layers.Dropout(dropout)

    def call(self, x, mask=None, training=False):
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]

        def split(t):
            t = tf.reshape(t, (B, T, self.h, self.d_k))
            return tf.transpose(t, [0, 2, 1, 3])

        Q, K, V = split(self.Wq(x)), split(self.Wk(x)), split(self.Wv(x))

        scores = tf.matmul(Q, K, transpose_b=True) / math.sqrt(self.d_k)
        if mask is not None:
            scores = tf.where(mask == 0, tf.constant(float("-inf"), dtype=scores.dtype), scores)
        attn = tf.nn.softmax(scores, axis=-1)
        attn = self.drop(attn, training=training)

        out = tf.matmul(attn, V)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, (B, T, self.embed_dim))
        return self.Wo(out)

# ─────────────────────────────────────────
# 5. DECODER BLOCK
# ─────────────────────────────────────────
class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadAttention(cfg.embed_dim, cfg.num_heads, cfg.dropout)
        self.ff1 = tf.keras.layers.Dense(cfg.ff_dim, activation="gelu")
        self.ff2 = tf.keras.layers.Dense(cfg.embed_dim)
        self.ff_drop = tf.keras.layers.Dropout(cfg.dropout)
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.drop = tf.keras.layers.Dropout(cfg.dropout)

    def call(self, x, mask=None, training=False):
        x = x + self.drop(self.attn(self.norm1(x), mask, training=training), training=training)
        ff_out = self.ff2(self.ff1(self.norm2(x)))
        ff_out = self.ff_drop(ff_out, training=training)
        x = x + ff_out
        return x

# ─────────────────────────────────────────
# 6. FULL DECODER MODEL (weight-tied head)
# ─────────────────────────────────────────
class KidnetGPT(tf.keras.Model):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = tf.keras.layers.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_emb = tf.keras.layers.Embedding(cfg.max_seq_len, cfg.embed_dim)
        self.drop = tf.keras.layers.Dropout(cfg.dropout)
        self.blocks = [DecoderBlock(cfg) for _ in range(cfg.num_layers)]
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, x, training=False):
        T = tf.shape(x)[1]
        pos = tf.range(T)[tf.newaxis, :]
        mask = tf.linalg.band_part(tf.ones((T, T)), -1, 0)[tf.newaxis, tf.newaxis, :, :]

        out = self.drop(self.tok_emb(x) + self.pos_emb(pos), training=training)
        for block in self.blocks:
            out = block(out, mask, training=training)
        out = self.norm(out)
        # weight-tied output head: reuse token embedding matrix as the projection
        logits = tf.matmul(out, self.tok_emb.embeddings, transpose_b=True)
        return logits

# ─────────────────────────────────────────
# 7. TRAINING LOOP
# ─────────────────────────────────────────
def masked_ce_loss(y_true, y_pred):
    mask = tf.cast(y_true != 0, tf.float32)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    loss = loss * mask
    return tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(mask), 1.0)


def train(model, dataset, optimizer, epoch):
    total_loss, n_steps = 0.0, 0

    @tf.function
    def step_fn(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = masked_ce_loss(y, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    for step, (x, y) in enumerate(dataset):
        loss = step_fn(x, y)
        total_loss += float(loss)
        n_steps += 1
        if step % 50 == 0:
            print(f"Epoch {epoch} | Step {step} | Loss {float(loss):.4f}")

    return total_loss / max(n_steps, 1)

# ─────────────────────────────────────────
# 8. GENERATION (top-k sampling)
# ─────────────────────────────────────────
def generate(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8, top_k=50):
    ids = [2] + tokenizer.encode(prompt)  # prepend <BOS>
    ids = tf.constant([ids], dtype=tf.int32)

    for _ in range(max_new_tokens):
        inp = ids[:, -cfg.max_seq_len:]
        logits = model(inp, training=False)
        logits = logits[:, -1, :] / temperature

        if top_k:
            top_vals, _ = tf.math.top_k(logits, k=top_k)
            threshold = top_vals[:, -1:]
            logits = tf.where(logits < threshold, tf.constant(float("-inf")), logits)

        next_tok = tf.random.categorical(logits, num_samples=1, dtype=tf.int32)
        ids = tf.concat([ids, next_tok], axis=1)

        if int(next_tok[0, 0]) == 3:  # <EOS>
            break

    return tokenizer.decode(ids[0].numpy().tolist()[1:])  # skip <BOS>

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
    ds = make_dataset(xs, ys, cfg.batch_size)
    steps_per_epoch = max(1, xs.shape[0] // cfg.batch_size)

    model = KidnetGPT(cfg)
    schedule = tf.keras.optimizers.schedules.CosineDecay(cfg.lr, cfg.epochs * steps_per_epoch)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=schedule)

    # build the model once to report parameter count
    dummy = tf.zeros((1, cfg.max_seq_len), dtype=tf.int32)
    _ = model(dummy, training=False)
    print(f"Model parameters: {model.count_params():,}")

    for epoch in range(1, cfg.epochs + 1):
        avg_loss = train(model, ds, optimizer, epoch)
        print(f"── Epoch {epoch} avg loss: {avg_loss:.4f}")

        if epoch % 5 == 0:
            out = generate(model, tokenizer, "the kidney", max_new_tokens=200)
            print(f"\nSample: {out}\n")

    model.save_weights("kidnet_gpt.weights.h5")
    print("Model saved.")

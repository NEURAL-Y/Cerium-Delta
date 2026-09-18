"""
Transformer QA model — JAX + Flax + Optax implementation.
Equivalent to the PyTorch version: encoder-decoder transformer trained
on (question, answer) pairs tokenized with bert-base-uncased.
"""

import math
import functools

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import train_state
from transformers import AutoTokenizer

# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

df = pd.read_csv(
    "https://raw.githubusercontent.com/NEURAL-Y/Cerium-Delta/main/test/jax_models/datasets/Gk_questions.csv"
)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

texts = df["Question"].tolist()
answers_text = df["Answer"].tolist()

MAX_LEN = 128

question_enc = tokenizer(
    texts, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="np"
)
answer_enc = tokenizer(
    answers_text, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="np"
)

src_ids = jnp.array(question_enc["input_ids"], dtype=jnp.int32)
tgt_ids = jnp.array(answer_enc["input_ids"], dtype=jnp.int32)

BATCH_SIZE = 16
N = src_ids.shape[0]


def data_loader(key, src, tgt, batch_size):
    perm = jax.random.permutation(key, src.shape[0])
    for i in range(0, len(perm) - batch_size + 1, batch_size):
        idx = perm[i:i + batch_size]
        yield src[idx], tgt[idx]


# ----------------------------------------------------------------------
# Model components
# ----------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    d_model: int
    num_heads: int

    def setup(self):
        assert self.d_model % self.num_heads == 0
        self.d_k = self.d_model // self.num_heads
        self.W_q = nn.Dense(self.d_model)
        self.W_k = nn.Dense(self.d_model)
        self.W_v = nn.Dense(self.d_model)
        self.W_o = nn.Dense(self.d_model)

    def split_heads(self, x):
        b, s, _ = x.shape
        return x.reshape(b, s, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

    def combine_heads(self, x):
        b, h, s, dk = x.shape
        return x.transpose(0, 2, 1, 3).reshape(b, s, h * dk)

    def __call__(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_scores = jnp.matmul(Q, jnp.swapaxes(K, -2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = jnp.where(mask == 0, -1e9, attn_scores)
        attn_probs = jax.nn.softmax(attn_scores, axis=-1)
        out = jnp.matmul(attn_probs, V)
        return self.W_o(self.combine_heads(out))


class PositionWiseFeedForward(nn.Module):
    d_model: int
    d_ff: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.d_ff)(x)
        x = nn.relu(x)
        return nn.Dense(self.d_model)(x)


def positional_encoding(max_seq_length, d_model):
    pe = np.zeros((max_seq_length, d_model), dtype=np.float32)
    position = np.arange(0, max_seq_length, dtype=np.float32)[:, None]
    div_term = np.exp(np.arange(0, d_model, 2).astype(np.float32) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return jnp.array(pe[None, :, :])


class EncoderLayer(nn.Module):
    d_model: int
    num_heads: int
    d_ff: int
    dropout: float

    @nn.compact
    def __call__(self, x, mask, train: bool):
        attn_out = MultiHeadAttention(self.d_model, self.num_heads)(x, x, x, mask)
        x = nn.LayerNorm()(x + nn.Dropout(self.dropout, deterministic=not train)(attn_out))
        ff_out = PositionWiseFeedForward(self.d_model, self.d_ff)(x)
        x = nn.LayerNorm()(x + nn.Dropout(self.dropout, deterministic=not train)(ff_out))
        return x


class DecoderLayer(nn.Module):
    d_model: int
    num_heads: int
    d_ff: int
    dropout: float

    @nn.compact
    def __call__(self, x, enc_output, src_mask, tgt_mask, train: bool):
        attn_out = MultiHeadAttention(self.d_model, self.num_heads)(x, x, x, tgt_mask)
        x = nn.LayerNorm()(x + nn.Dropout(self.dropout, deterministic=not train)(attn_out))
        attn_out = MultiHeadAttention(self.d_model, self.num_heads)(x, enc_output, enc_output, src_mask)
        x = nn.LayerNorm()(x + nn.Dropout(self.dropout, deterministic=not train)(attn_out))
        ff_out = PositionWiseFeedForward(self.d_model, self.d_ff)(x)
        x = nn.LayerNorm()(x + nn.Dropout(self.dropout, deterministic=not train)(ff_out))
        return x


class Transformer(nn.Module):
    src_vocab_size: int
    tgt_vocab_size: int
    d_model: int
    num_heads: int
    num_layers: int
    d_ff: int
    max_seq_length: int
    dropout: float

    def setup(self):
        self.encoder_embedding = nn.Embed(self.src_vocab_size, self.d_model)
        self.decoder_embedding = nn.Embed(self.tgt_vocab_size, self.d_model)
        self.pe = positional_encoding(self.max_seq_length, self.d_model)
        self.encoder_layers = [
            EncoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout)
            for _ in range(self.num_layers)
        ]
        self.decoder_layers = [
            DecoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout)
            for _ in range(self.num_layers)
        ]
        self.fc = nn.Dense(self.tgt_vocab_size)
        self.dropout_layer = nn.Dropout(self.dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0)[:, None, None, :]
        tgt_pad_mask = (tgt != 0)[:, None, :, None]
        seq_len = tgt.shape[1]
        nopeak = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        tgt_mask = tgt_pad_mask & nopeak[None, None, :, :]
        return src_mask, tgt_mask

    def __call__(self, src, tgt, train: bool = True):
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        src_emb = self.encoder_embedding(src) + self.pe[:, : src.shape[1]]
        src_emb = self.dropout_layer(src_emb, deterministic=not train)
        tgt_emb = self.decoder_embedding(tgt) + self.pe[:, : tgt.shape[1]]
        tgt_emb = self.dropout_layer(tgt_emb, deterministic=not train)

        enc_out = src_emb
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_mask, train=train)

        dec_out = tgt_emb
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, src_mask, tgt_mask, train=train)

        return self.fc(dec_out)


# ----------------------------------------------------------------------
# Train state / loss / step
# ----------------------------------------------------------------------

VOCAB_SIZE = len(tokenizer)

model = Transformer(
    src_vocab_size=VOCAB_SIZE,
    tgt_vocab_size=VOCAB_SIZE,
    d_model=512,
    num_heads=8,
    num_layers=6,
    d_ff=2048,
    max_seq_length=MAX_LEN,
    dropout=0.1,
)

rng = jax.random.PRNGKey(0)
rng, init_rng, dropout_rng = jax.random.split(rng, 3)
dummy_src = jnp.ones((BATCH_SIZE, MAX_LEN), dtype=jnp.int32)
dummy_tgt = jnp.ones((BATCH_SIZE, MAX_LEN - 1), dtype=jnp.int32)
params = model.init({"params": init_rng, "dropout": dropout_rng}, dummy_src, dummy_tgt)["params"]

tx = optax.adam(learning_rate=1e-4, b1=0.9, b2=0.98, eps=1e-9)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def loss_fn(params, src, tgt_in, tgt_out, dropout_rng):
    logits = model.apply({"params": params}, src, tgt_in, train=True, rngs={"dropout": dropout_rng})
    mask = (tgt_out != 0).astype(jnp.float32)
    one_hot = jax.nn.one_hot(tgt_out, VOCAB_SIZE)
    loss_per_token = optax.softmax_cross_entropy(logits, one_hot)
    return (loss_per_token * mask).sum() / jnp.maximum(mask.sum(), 1.0)


@jax.jit
def train_step(state, src, tgt, dropout_rng):
    tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
    dropout_rng, step_rng = jax.random.split(dropout_rng)
    loss, grads = jax.value_and_grad(loss_fn)(state.params, src, tgt_in, tgt_out, step_rng)
    state = state.apply_gradients(grads=grads)
    return state, loss, dropout_rng


# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------

for epoch in range(100):
    rng, perm_rng = jax.random.split(rng)
    epoch_loss = None
    for src_batch, tgt_batch in data_loader(perm_rng, src_ids, tgt_ids, BATCH_SIZE):
        state, epoch_loss, dropout_rng = train_step(state, src_batch, tgt_batch, dropout_rng)
    if epoch % 10 == 0:
        print(f"Epoch: {epoch + 1}, Loss: {epoch_loss:.4f}")


# ----------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------

def predict(params, question, tokenizer, max_length=MAX_LEN):
    inputs = tokenizer(
        question, padding="max_length", truncation=True, max_length=max_length, return_tensors="np"
    )
    src = jnp.array(inputs["input_ids"], dtype=jnp.int32)
    tgt = jnp.array([[tokenizer.cls_token_id]], dtype=jnp.int32)

    for _ in range(max_length):
        logits = model.apply({"params": params}, src, tgt, train=False)
        next_token = jnp.argmax(logits[:, -1, :], axis=-1)[:, None]
        tgt = jnp.concatenate([tgt, next_token], axis=1)
        if int(next_token[0, 0]) == tokenizer.sep_token_id:
            break

    return tokenizer.decode(np.array(tgt[0]), skip_special_tokens=True)


if __name__ == "__main__":
    test_questions = [
        "What is the capital of France?",
        "How does photosynthesis work?",
        "Who is the author of 'The Theory of Relativity'?",
        "Which instrument has 88 keys?",
        "'Harry Potter'?",
    ]
    print("\nTesting the model with some questions:")
    for q in test_questions:
        print(f"\nQ: {q}")
        print(f"A: {predict(state.params, q, tokenizer)}")

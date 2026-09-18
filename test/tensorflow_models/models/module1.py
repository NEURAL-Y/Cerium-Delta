"""
Transformer QA model — TensorFlow + Keras implementation.
Equivalent to the PyTorch version: encoder-decoder transformer trained
on (question, answer) pairs tokenized with bert-base-uncased.
"""

import math

import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer

# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

df = pd.read_csv(
    "https://raw.githubusercontent.com/NEURAL-Y/Cerium-Delta/main/test/tensorflow_models/datasets/Gk_questions.csv"
)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

texts = df["Question"].tolist()
answers_text = df["Answer"].tolist()

MAX_LEN = 128

question_enc = tokenizer(
    texts, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="tf"
)
answer_enc = tokenizer(
    answers_text, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="tf"
)

src_ids = question_enc["input_ids"]
tgt_ids = answer_enc["input_ids"]

BATCH_SIZE = 16
dataset = tf.data.Dataset.from_tensor_slices((src_ids, tgt_ids))
dataset = dataset.shuffle(buffer_size=len(texts)).batch(BATCH_SIZE, drop_remainder=True)

VOCAB_SIZE = len(tokenizer)

gpus = tf.config.list_physical_devices("GPU")
print(f"Using device: {'GPU' if gpus else 'CPU'}")


# ----------------------------------------------------------------------
# Model components
# ----------------------------------------------------------------------

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = tf.keras.layers.Dense(d_model)
        self.W_k = tf.keras.layers.Dense(d_model)
        self.W_v = tf.keras.layers.Dense(d_model)
        self.W_o = tf.keras.layers.Dense(d_model)

    def split_heads(self, x):
        b = tf.shape(x)[0]
        s = tf.shape(x)[1]
        x = tf.reshape(x, (b, s, self.num_heads, self.d_k))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def combine_heads(self, x):
        b = tf.shape(x)[0]
        s = tf.shape(x)[2]
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(x, (b, s, self.d_model))

    def call(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_scores = tf.matmul(Q, K, transpose_b=True) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = tf.where(mask == 0, tf.constant(-1e9, dtype=attn_scores.dtype), attn_scores)
        attn_probs = tf.nn.softmax(attn_scores, axis=-1)
        out = tf.matmul(attn_probs, V)
        return self.W_o(self.combine_heads(out))


class PositionWiseFeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(d_ff, activation="relu")
        self.fc2 = tf.keras.layers.Dense(d_model)

    def call(self, x):
        return self.fc2(self.fc1(x))


def positional_encoding(max_seq_length, d_model):
    pe = np.zeros((max_seq_length, d_model), dtype=np.float32)
    position = np.arange(0, max_seq_length, dtype=np.float32)[:, None]
    div_term = np.exp(np.arange(0, d_model, 2).astype(np.float32) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return tf.constant(pe[None, :, :], dtype=tf.float32)


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, mask, training):
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out, training=training))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out, training=training))
        return x


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.norm3 = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, enc_output, src_mask, tgt_mask, training):
        attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_out, training=training))
        attn_out = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_out, training=training))
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out, training=training))
        return x


class Transformer(tf.keras.Model):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads,
                 num_layers, d_ff, max_seq_length, dropout):
        super().__init__()
        self.encoder_embedding = tf.keras.layers.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = tf.keras.layers.Embedding(tgt_vocab_size, d_model)
        self.pe = positional_encoding(max_seq_length, d_model)

        self.encoder_layers = [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        self.decoder_layers = [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]

        self.fc = tf.keras.layers.Dense(tgt_vocab_size)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = tf.cast(src != 0, tf.float32)[:, tf.newaxis, tf.newaxis, :]
        tgt_pad_mask = tf.cast(tgt != 0, tf.float32)[:, tf.newaxis, :, tf.newaxis]
        seq_len = tf.shape(tgt)[1]
        nopeak = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        tgt_mask = tgt_pad_mask * nopeak[tf.newaxis, tf.newaxis, :, :]
        return src_mask, tgt_mask

    def call(self, inputs, training=False):
        src, tgt = inputs
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        src_emb = self.encoder_embedding(src) + self.pe[:, : tf.shape(src)[1]]
        src_emb = self.dropout(src_emb, training=training)
        tgt_emb = self.decoder_embedding(tgt) + self.pe[:, : tf.shape(tgt)[1]]
        tgt_emb = self.dropout(tgt_emb, training=training)

        enc_out = src_emb
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_mask, training=training)

        dec_out = tgt_emb
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, src_mask, tgt_mask, training=training)

        return self.fc(dec_out)


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

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

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")


def masked_loss(y_true, y_pred):
    mask = tf.cast(y_true != 0, tf.float32)
    loss = loss_object(y_true, y_pred) * mask
    return tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(mask), 1.0)


@tf.function
def train_step(src_batch, tgt_batch):
    tgt_in = tgt_batch[:, :-1]
    tgt_out = tgt_batch[:, 1:]
    with tf.GradientTape() as tape:
        logits = model((src_batch, tgt_in), training=True)
        loss = masked_loss(tgt_out, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


for epoch in range(100):
    epoch_loss = None
    for src_batch, tgt_batch in dataset:
        epoch_loss = train_step(src_batch, tgt_batch)
    if epoch % 10 == 0:
        print(f"Epoch: {epoch + 1}, Loss: {epoch_loss.numpy():.4f}")


# ----------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------

def predict(model, question, tokenizer, max_length=MAX_LEN):
    inputs = tokenizer(
        question, padding="max_length", truncation=True, max_length=max_length, return_tensors="tf"
    )
    src = inputs["input_ids"]
    tgt = tf.constant([[tokenizer.cls_token_id]], dtype=tf.int32)

    for _ in range(max_length):
        logits = model((src, tgt), training=False)
        next_token = tf.argmax(logits[:, -1, :], axis=-1, output_type=tf.int32)[:, tf.newaxis]
        tgt = tf.concat([tgt, next_token], axis=1)
        if int(next_token[0, 0]) == tokenizer.sep_token_id:
            break

    return tokenizer.decode(tgt[0].numpy(), skip_special_tokens=True)


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
        print(f"A: {predict(model, q, tokenizer)}")

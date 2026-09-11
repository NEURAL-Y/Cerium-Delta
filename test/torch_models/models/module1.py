import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy
from transformers import AutoTokenizer
import math
import torch
from torch.utils.data import Dataset, DataLoader

df= pd.read_csv("https://raw.githubusercontent.com/NEURAL-Y/Cerium-Delta/main/test/torch_models/datasets/Gk_questions.csv")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# Using a pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# Tokenize all texts
texts = df["Question"].tolist()
text= df["Answer"].tolist()

# Tokenize and convert to input IDs and attention masks
question = tokenizer(texts, 
                      padding=True,      # Pad sequences to max length
                      truncation=True,   # Truncate long sequences
                      max_length=128,    # Optional: max tokens per sentence
                      return_tensors="pt") # PyTorch tensors
answer = tokenizer(text,
                        padding=True,      # Pad sequences to max length
                        truncation=True,   # Truncate long sequences
                        max_length=128,    # Optional: max tokens per sentence
                        return_tensors="pt") # PyTorch tensors

class QADataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels["input_ids"]  # Target token IDs
        
    def __len__(self):
        return len(self.encodings["input_ids"])
    
    def __getitem__(self, idx):
        item = {key: val[idx].to(device) for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx].to(device)
        return item

dataset = QADataset(question, answer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]#type:ignore
    

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
    
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        device = src.device  # Get the device from input tensor
        
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        
        seq_length = tgt.size(1)
        # Create nopeak_mask on the same device as src/tgt
        nopeak_mask = (1 - torch.triu(
            torch.ones(1, seq_length, seq_length, device=device),
            diagonal=1
        )).bool()
        
        # Both masks are now on the same device
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output


# Move tokenized data to GPU
question = {k: v.to(device) for k, v in question.items()}
answer = {k: v.to(device) for k, v in answer.items()}

model = Transformer(
    src_vocab_size=len(tokenizer),
    tgt_vocab_size=len(tokenizer),
    d_model=512,
    num_heads=8,
    num_layers=6,   
    d_ff=2048,
    max_seq_length=128,
    dropout=0.1
).to(device)    # Add .to(device) here
for name,weights in model.named_parameters():
    print(name,weights.detach().cpu().numpy())
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

model.train()
for epoch in range(100):
    for batch in dataloader:
        src_data = batch["input_ids"]  # Already on GPU from dataset
        tgt_data = batch["labels"]     # Already on GPU from dataset
        
        optimizer.zero_grad()
        output = model(src_data, tgt_data[:, :-1])
        
        loss = criterion(
            output.contiguous().view(-1, len(tokenizer)), 
            tgt_data[:, 1:].contiguous().view(-1)
        )
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:  # Print less frequently
            print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

# Modify evaluation
model.eval()
def predict(model, question, tokenizer, max_length=128, device='cuda'):
    model.eval()
    
    # Tokenize the question
    inputs = tokenizer(
        question,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)

    # Initialize target with start token
    tgt = torch.tensor([[tokenizer.cls_token_id]]).to(device)
    
    with torch.no_grad():#type:ignore
        for _ in range(max_length):
            # Get prediction
            output = model(inputs["input_ids"], tgt)
            
            # Get next token prediction
            next_token_logits = output[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
            
            # Add predicted token to target sequence
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # Stop if we predict the end token
            if next_token.item() == tokenizer.sep_token_id:
                break
    
    # Decode the generated answer
    answer = tokenizer.decode(tgt[0], skip_special_tokens=True)
    return answer

# Example usage
if __name__ == "__main__":
    # Test the model with some questions
    test_questions = [
        "What is the capital of France?",
        "How does photosynthesis work?",
        "Who is the author of 'The Theory of Relativity'?",
        "Which instrument has 88 keys?",
        " 'Harry Potter'?"
    ]
    
    print("\nTesting the model with some questions:")
    for question in test_questions:
        print(f"\nQ: {question}")
        answer = predict(model, question, tokenizer, device=device)#type:ignore
        print(f"A: {answer}")

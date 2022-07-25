import torch
import torch.nn as nn
import math


class Embedding(nn.Module):
    def __init__(self, vocab_size, model_dimension):
        super().__init__()
        self.embeddings_table = nn.Embedding(vocab_size, model_dimension)
        self.model_dimension = model_dimension

    def forward(self, token_ids_batch):
        assert token_ids_batch.ndim == 2, f'Expected: (batch size, max token sequence length), got {token_ids_batch.shape}'
        embeddings = self.embeddings_table(token_ids_batch)
        return embeddings * math.sqrt(self.model_dimension)


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim=512, n_heads=8) -> None:
        super().__init__()
        assert input_dim % n_heads == 0, "Input dimension %d is not evenly divisible by n_heads %d" % (input_dim, n_heads)
        self.n_heads = n_heads
        self.head_dim = input_dim // n_heads
        self.key_proj = nn.Linear(input_dim, input_dim)
        self.query_proj = nn.Linear(input_dim, input_dim)
        self.value_proj = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.out_proj = nn.Linear(input_dim, input_dim)

    def forward(self, keys: torch.Tensor, queries: torch.Tensor, values: torch.Tensor, mask: torch.Tensor = None):
        # x - [bs, seq_len, dim]
        bs, seq_len = keys.shape[0], keys.shape[1]
        queries_seq_len = queries.shape[1]
        keys = self.key_proj(keys).view(bs, seq_len, self.n_heads, self.head_dim)
        queries = self.query_proj(queries).view(bs, queries_seq_len, self.n_heads, self.head_dim)
        values = self.value_proj(values).view(bs, seq_len, self.n_heads, self.head_dim)

        out, attn = self.attention(keys, queries, values, mask)

        out = out.transpose(2, 1).reshape(bs, queries_seq_len, self.n_heads * self.head_dim)
        return self.out_proj(out)

    def attention(self, keys: torch.Tensor, queries: torch.Tensor, values: torch.Tensor, mask: torch.Tensor = None):
        # keys - [bs, seq_len, n_heads, head_dim]
        keys = keys.transpose(2, 1)
        queries = queries.transpose(2, 1)
        values = values.transpose(2, 1)

        # scores - [bs, n_heads, seqlen, seqlen]
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores.masked_fill_(mask == torch.tensor(False), float("-inf"))

        attn = self.softmax(scores)

        # Aggregate values accross the sequence
        out = torch.matmul(attn, values)    
        return out, attn


class PointWiseNet(nn.Module):
    def __init__(self, input_dim=512, internal_dim=2048, dropout_p=0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_dim, internal_dim)
        self.linear2 = nn.Linear(internal_dim, input_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        return self.linear2(x)


class EncoderBlock(nn.Module):
    def __init__(self, dim=512, n_heads=8, pw_net_dim=2048, dropout_p=0.1) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(dim, n_heads)
        self.pw_net = PointWiseNet(dim, pw_net_dim)
        self.norm_layer = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor, mask: torch.Tensor=None):
        out = self.attention(x, x, x, mask)
        x = self.norm_layer(x + out)
        out = self.pw_net(x)
        out = self.dropout(out)
        return self.norm_layer(x + out)
    
    
class DecoderBlock(nn.Module):
    def __init__(self, dim=512, n_heads=8, pw_net_dim=2048, dropout_p=0.1) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(dim, n_heads)
        self.cross_attention = MultiHeadAttention(dim, n_heads)
        self.pw_net = PointWiseNet(dim, pw_net_dim, dropout_p=dropout_p)
        self.norm_layer = nn.LayerNorm(dim)

    def forward(self, x, enc_out, mask=None, cross_mask=None):
        out = self.self_attention(x, x, x, mask)
        x = self.norm_layer(x + out)

        out = self.cross_attention(enc_out, x, enc_out, cross_mask)
        x = self.norm_layer(x + out)

        out = self.pw_net(x)
        return self.norm_layer(x + out)
    

class Encoder(nn.Module):
    def __init__(self, dim=512, n_heads=8, pw_net_dim=2048, n_blocks=6, dropout_p=0.1) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([EncoderBlock(dim, n_heads, pw_net_dim, dropout_p) for _ in range(n_blocks)])

    def forward(self, x, mask=None):
        for l in self.blocks:
            x = l(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, dim=512, n_heads=8, pw_net_dim=2048, n_blocks=6, dropout_p=0.1) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([DecoderBlock(dim, n_heads, pw_net_dim, dropout_p) for _ in range(n_blocks)])

    def forward(self, x, enc_out, mask=None, cross_mask=None):
        for l in self.blocks:
            x = l(x, enc_out, mask, cross_mask)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, dim=512, dropout_p=0.1) -> None:
        super().__init__()
        self.dim = dim
        self.embedding = nn.Parameter(torch.randn(1, seq_len, dim))
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        seq_len = x.shape[1]
        return self.dropout(x + self.embedding[:, :seq_len] * math.sqrt(self.dim))
    
    
class Transformer(nn.Module):
    def __init__(self, seq_len, src_vocab_size, trg_vocab_size, dim=512, n_heads=8, pw_net_dim=2048, n_en_blocks=6, n_de_blocks=6, dropout_p=0.1) -> None:
        super().__init__()
        self.src_embedding = Embedding(src_vocab_size, dim)
        self.trg_embedding = Embedding(trg_vocab_size, dim)
        self.src_pos_encoding = PositionalEncoding(seq_len, dropout_p=dropout_p)
        self.trg_pos_encoding = PositionalEncoding(seq_len, dropout_p=dropout_p)
        self.encoder = Encoder(dim, n_heads, pw_net_dim, n_en_blocks, dropout_p=dropout_p)
        self.decoder = Decoder(dim, n_heads, pw_net_dim, n_de_blocks, dropout_p=dropout_p)

        self.head = nn.Linear(dim, trg_vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.init_params()

    def forward(self, src_seq_indices, trg_seq_indices, src_mask = None, trg_mask = None):
        src_seq = self.src_embedding(src_seq_indices)
        trg_seq = self.trg_embedding(trg_seq_indices)

        src_seq = self.src_pos_encoding(src_seq)
        trg_seq = self.trg_pos_encoding(trg_seq)

        enc_out = self.encoder(src_seq, src_mask)
        dec_out = self.decoder(trg_seq, enc_out, trg_mask, src_mask)
        return enc_out, self.log_softmax(self.head(dec_out))
    
    def decoder_forward(self, trg_seq_indices, enc_out, trg_mask, src_mask):
        trg_seq = self.trg_embedding(trg_seq_indices)
        trg_seq = self.trg_pos_encoding(trg_seq)
        dec_out = self.decoder(trg_seq, enc_out, trg_mask, src_mask)
        return self.log_softmax(self.head(dec_out))

    def init_params(self, default_initialization=False):
        if not default_initialization:
            for name, p in self.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)


if __name__ == '__main__':
    model = Transformer(seq_len=128, src_vocab_size=128, trg_vocab_size=128)
    input_seq = torch.randint(0, 128, size=(32, 128))
    target_seq = torch.randint(0, 128, size=(32, 128))
    model(input_seq, target_seq)
    print('It works!')

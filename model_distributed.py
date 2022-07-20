from model import Embedding, PositionalEncoding, Encoder, Decoder
import torch.nn as nn

"""
This file contains exactly the same transformer implementation but with some
tweaks to make distributed training possible. Specifically, the loss computation
is made to be a part of the model because the output tensor is too large to fit
on a single GPU.
"""


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

        self.is_data_parallel = False
    
    def set_data_parallel(self, is_data_parallel):
        self.is_data_parallel = is_data_parallel

    def forward(self, src_seq_indices, trg_seq_indices, src_mask=None, trg_mask=None, criterion=None, labels=None):
        if self.is_data_parallel:
            return self.compute_loss(src_seq_indices, trg_seq_indices, src_mask, trg_mask, criterion, labels)
        else:
            return self.forward_(src_seq_indices, trg_seq_indices, src_mask, trg_mask)

    def forward_(self, src_seq_indices, trg_seq_indices, src_mask=None, trg_mask=None):
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

    # Used in DataParallel
    def compute_loss(self, src_seq, trg_input_seq, src_mask, trg_mask, criterion, labels):
        enc_out, logits = self.forward_(src_seq, trg_input_seq, src_mask, trg_mask)
        return criterion(logits.transpose(1, 2), labels)

    def init_params(self, default_initialization=False):
        # Not mentioned in the paper, but other implementations used xavier.
        # I tested both PyTorch's default initialization and this, and xavier has tremendous impact! I didn't expect
        # a model's perf, with normalization layers, to be so much dependent on the choice of weight initialization.
        if not default_initialization:
            for name, p in self.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
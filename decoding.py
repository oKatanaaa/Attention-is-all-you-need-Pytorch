import torch
from data_utils import generate_mask_src, generate_mask_trg


class GreedyDecoder:
    def __init__(self, model: torch.nn.Module, tokenizer, src_vocab, trg_vocab, eos_token_id, sos_token_id, pad_token_id, max_seq_length=128) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.eos_token_id = eos_token_id
        self.sos_token_id = sos_token_id
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length

    def decode(self, sentence):
        if sentence == '':
            return ''

        self.model.eval()

        tokens = self.tokenizer(sentence.lower())
        token_ids = self.src_vocab(tokens)[:self.max_seq_length]

        target_input = [self.sos_token_id]
        first_iteration = True
        enc_out = None
        with torch.no_grad():
            # Decode until you encounter EOS token or hit the length limit.
            while True:
                input_seq = torch.tensor([token_ids], device=next(iter(self.model.parameters())).device)
                target_seq = torch.tensor([target_input], device=next(iter(self.model.parameters())).device)

                # You may not necessarily generate src_mask, but trg_mask is required to mask out previous tokens.
                src_mask = generate_mask_src(input_seq, self.pad_token_id)
                trg_mask = generate_mask_trg(target_seq, self.pad_token_id)

                if first_iteration:
                    enc_out, logits = self.model(input_seq, target_seq, src_mask, trg_mask)
                    first_iteration = False
                else:
                    logits = self.model.decoder_forward(target_seq, enc_out, trg_mask, src_mask)

                last_token_id = logits[0, -1].argmax()

                if last_token_id == self.eos_token_id or len(target_input) == self.max_seq_length - 1:
                    break

                target_input.append(last_token_id)
        
        return ' '.join([self.trg_vocab.get_itos()[ind] for ind in target_input[1:]])
        
        

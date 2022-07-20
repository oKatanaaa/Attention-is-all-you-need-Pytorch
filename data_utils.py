from torchtext.vocab import vocab, Vocab
from torchtext.transforms import PadTransform
from collections import Counter, OrderedDict
import os
import pickle
import torch
from sklearn.utils import shuffle
from tqdm import tqdm
from typing import Iterable, Callable, Union


class Specials:
    PAD = '<PAD>'
    SOS = '<SOS>'
    EOS = '<EOS>'
    UNK = '<UNK>'


def make_vocab(text_iter, tokenizer, specials, min_freq=5, voc_cache_name='vocab.pkl', cache=True, force_recreate=False):
    if os.path.exists(voc_cache_name) and not force_recreate:
        with open(voc_cache_name, 'rb') as file:
            vocab_obj = pickle.load(file)
        return vocab_obj

    counter = Counter()
    for line in tqdm(text_iter):
        counter.update([token.text for token in tokenizer(line.lower())])
    
    def construct_vocab(counter, specials):
        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        ordered_dict = OrderedDict(sorted_by_freq_tuples)
        vocab_obj = vocab(ordered_dict, min_freq=min_freq, specials=specials)
        vocab_obj.set_default_index(vocab_obj[Specials.UNK])
        return vocab_obj
    
    def save_vocab(vocab_obj, filename):
        import pickle
        with open(filename, 'wb') as file:
            pickle.dump(vocab_obj, file)

    vocab_obj = construct_vocab(counter, specials=specials)
    if cache:
        save_vocab(vocab_obj, voc_cache_name)
    return vocab_obj


class TextPadderDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, src_vocab: Vocab, trg_vocab: Vocab, text_iterator: Iterable, 
        src_tokenizer_fn: Callable, trg_tokenizer_fn: Callable,
        device: Union[str, torch.device], max_seq_length: int,
        pad_token=Specials.PAD, use_cache=True
    ):
        """
        A simple dataset that generates padded sequences of token indices of fixed length.

        Parameters
        ----------
        src_vocab : Vocab
            Source language vocabulary.
        trg_vocab : Vocab
            Target language vocabulary.
        text_iterator : Iterable
            An iterable yielding pairs of sentences (str: source, str: target)
        src_tokenizer_fn : Callable
            A callable that takes in a source sentence and returns a list of individual words.
        trg_tokenizer_fn : Callable
            A callable that takes in a target sentence and returns a list of individual words.
        device : Union[str, torch.device]
            Device where to place indeces tensor on.
        max_seq_length : int
            Maximum sequence length. Sentences of length larger than that are skipped.
        pad_token : str, optional
            The padding token, by default Specials.PAD
        use_cache : bool, optional
            Whether to cache the dataset, by default True.
            NOTE: this accelerates training significantly after the first epoch,
            though this pipeline is already fast enough.
        """
        assert src_vocab[pad_token] == trg_vocab[pad_token], 'Padding token must have the same index across vocabs.'
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.text_iterator = text_iterator
        self.src_tokenizer_fn = src_tokenizer_fn
        self.trg_tokenizer_fn = trg_tokenizer_fn
        self.max_seq_length = max_seq_length
        self.debug = False
        self.device = device
        self.pad_transform = PadTransform(max_length=max_seq_length, pad_value=pad_token)
        self.cache = [] if use_cache else None
        self.is_cache_ready = False
            
    def set_debug(self, debug):
        self.debug = debug

    def __iter__(self):
        if self.is_cache_ready:
            self.cache = shuffle(self.cache)
            for trg_token_ids, src_token_ids in self.cache:
                yield trg_token_ids, src_token_ids
        else:
            for tgt, src in self.text_iterator:
                src_tokens = self.src_tokenizer_fn(src)
                trg_tokens = [Specials.SOS] + self.src_tokenizer_fn(tgt) + [Specials.EOS]
                
                if self.debug:
                    yield trg_tokens, src_tokens

                if len(src_tokens) == 0 or len(trg_tokens) > self.max_seq_length or len(src_tokens) > self.max_seq_length:
                    # The sequence is degenerate, skip it.
                    # NOTE: There is only one empty sequence at the end of the dataset.
                    continue

                src_token_ids = torch.tensor(self.src_vocab(src_tokens), device=self.device)
                trg_token_ids = torch.tensor(self.trg_vocab(trg_tokens), device=self.device)
                src_token_ids = self.pad_transform(src_token_ids)
                trg_token_ids = self.pad_transform(trg_token_ids)

                if self.cache is not None:
                    self.cache.append((trg_token_ids, src_token_ids))

                yield trg_token_ids, src_token_ids

            # Generator is exhausted, set cache ready flag to True.
            if self.cache is not None:
                self.is_cache_ready = True
    

def generate_mask_src(seq, pad_token_id):
    # [bs, seq_len]
    bs = seq.shape[0]
    src_mask = (seq != pad_token_id).view(bs, 1, 1, -1)
    return src_mask


def generate_mask_trg(seq, pad_token_id):
    # [bs, seq_len]
    bs, seq_len = seq.shape[0], seq.shape[1]
    pad_mask = (seq != pad_token_id).view(bs, 1, 1, -1)
    tri_mask = torch.triu(torch.ones(seq_len, seq_len, device=seq.device) == 1).transpose(0, 1).view(1, 1, seq_len, seq_len)
    trg_mask = pad_mask & tri_mask
    return trg_mask

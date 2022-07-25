import torch
from model import Transformer
from data_utils import generate_mask_src, generate_mask_trg, make_vocab
from data_utils import TextPadderDataset, Specials
from torch.utils.data import DataLoader
from datasets import YandexDataset
import spacy
from tqdm import tqdm
import os
from nltk.translate.bleu_score import corpus_bleu


BATCH_SIZE = 48
SEQ_LEN = 64
EPOCHS = 5
PAD_TOKEN_ID = 1
WEIGHTS_NAME = 'weights.pth'


def calculate_bleu_score(decoder, dataloader, src_preprocess_fn, trg_preprocess_fn, max_len):
    preds = []
    targets = []
    for trg, src in tqdm(list(dataloader)):
        if len(trg_preprocess_fn(trg)) >= max_len - 1 or len(src) == 0 or len(src_preprocess_fn(src)) >= max_len:
            continue

        out = decoder.decode(src)
        preds.append(trg_preprocess_fn(out))
        targets.append([trg_preprocess_fn(trg)])

    bleu_score = corpus_bleu(targets, preds)
    print(f'BLEU-4 corpus score = {bleu_score}, corpus length = {len(targets)}.')
    return bleu_score


def create_dataloader(batch_size=32, seq_len=64, device='cpu'):
    train_iter, valid_iter, test_iter = YandexDataset('datasets/Yandex').get_iters() # Multi30k('./data')
    
    en_tokenizer = spacy.load("en_core_web_sm")
    ru_tokenizer = spacy.load('ru_core_news_sm')
    # Disable all of the unnecessary pipes to accelerate the data pipeline.
    en_tokenizer.disable_pipes(['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner'])
    ru_tokenizer.disable_pipes(['tok2vec', 'morphologizer', 'parser', 'attribute_ruler', 'lemmatizer', 'ner'])
    
    en_vocab = make_vocab(
        text_iter=[trg for trg, src in train_iter], 
        tokenizer=en_tokenizer, 
        specials=(Specials.UNK, Specials.PAD, Specials.SOS, Specials.EOS), 
        min_freq=5,
        voc_cache_name='en_vocab.pkl'
    )
    ru_vocab = make_vocab(
        text_iter=[src for trg, src in train_iter], 
        tokenizer=ru_tokenizer, 
        min_freq=10, 
        specials=(Specials.UNK, Specials.PAD), 
        voc_cache_name='ru_vocab.pkl'
    )
    
    my_dataset = TextPadderDataset(
        ru_vocab, en_vocab,
        train_iter,
        lambda line: [token.text for token in ru_tokenizer(line.lower())],
        lambda line: [token.text for token in en_tokenizer(line.lower())],
        max_seq_length=seq_len,
        device=device
    )
    
    loader = DataLoader(my_dataset, batch_size=batch_size)
    return loader, valid_iter, en_vocab, ru_vocab, en_tokenizer, ru_tokenizer


def training_loop(model, dataloader, criterion, optim, epochs):
    device = next(iter(model.parameters())).device
    for epoch in range(epochs):
        model.train()
        iterator = tqdm(enumerate(dataloader))
        for i, (trg_seq, src_seq) in iterator:
            optim.zero_grad()

            trg_input_seq = trg_seq[:, :-1]
            trg_label_seq = trg_seq[:, 1:]
            trg_input_seq = trg_input_seq.to(device); src_seq = src_seq.to(device)
            trg_label_seq = trg_label_seq.to(device)

            src_mask = generate_mask_src(src_seq, PAD_TOKEN_ID)
            trg_mask = generate_mask_trg(trg_input_seq, PAD_TOKEN_ID)

            enc_out, logits = model(src_seq, trg_input_seq, src_mask, trg_mask)
            loss = criterion(logits.transpose(1, 2), trg_label_seq) / float(src_seq.shape[0])

            loss.backward()
            optim.step()
            iterator.set_postfix_str(f"Epoch:{epoch}. Iteration: {i}. Loss: {loss}")
        print(f"Epoch:{epoch}. Loss: {loss}")


def main(batch_sz, seq_len, epochs, pretrained_weghts=None, gpu_id=0):
    print(f'Batch size: {batch_sz}, seq_len: {seq_len}, epochs: {epochs}, gpu_id: {gpu_id}')
    print('Creating dataloader...')
    device = f'cuda:{gpu_id}'
    dataloader, valid_iter, en_vocab, de_vocab, trg_tokenizer, src_tokenizer = create_dataloader(batch_size=batch_sz, seq_len=seq_len, device=device)

    print('Creating model...')
    model = Transformer(seq_len=seq_len, src_vocab_size=len(de_vocab), trg_vocab_size=len(en_vocab))
    model.to(device)

    if pretrained_weghts is not None:
        print('Received a path to pretrained weights: {pretrained_weghts}.')
        model.load_state_dict(torch.load(pretrained_weghts), strict=False)
        print('Successfully loaded pretrained weights.')

    print('Initializing training dependencies...')
    optim = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    loss_obj = torch.nn.CrossEntropyLoss(reduction='sum', label_smoothing=0.1, ignore_index=PAD_TOKEN_ID)

    try:
        print('Start training...')
        training_loop(model, dataloader, loss_obj, optim, epochs)
        print('Training has ended.')
    except Exception as ex:
        print(ex)

    return model


from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-bs', '--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('-e', '--epochs', type=int, default=EPOCHS)
    parser.add_argument('-sl', '--seq_len', type=int, default=SEQ_LEN)
    parser.add_argument('-s', '--save', type=str, default='./weights/result', help='Saving folder')
    parser.add_argument('-pw', '--pretrained_weights', type=str, default=None, help='Path to pretrained weights file (.pth).')
    parser.add_argument('--gpu_id', type=int, default=0)

    args = parser.parse_args()
    
    model = main(args.batch_size, args.seq_len, args.epochs, args.pretrained_weights, args.gpu_id)

    print('Saving the model...')
    os.makedirs(args.save, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.save, WEIGHTS_NAME))
    print('Model is saved.')

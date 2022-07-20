import torch
from model_distributed import Transformer
from data_utils import generate_mask_src, generate_mask_trg
from train import create_dataloader
from tqdm import tqdm
import os


BATCH_SIZE = 48
SEQ_LEN = 32
EPOCHS = 5
PAD_TOKEN_ID = 1
WEIGHTS_NAME = 'weights.pth'


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

            loss = model(src_seq, trg_input_seq, src_mask, trg_mask, criterion, trg_label_seq).sum() / float(src_seq.shape[0])

            loss.backward()
            optim.step()
            iterator.set_postfix_str(f"Epoch:{epoch}. Iteration: {i}. Loss: {loss}")
        print(f"Epoch:{epoch}. Loss: {loss}")


def main(batch_sz, seq_len, epochs, pretrained_weghts=None, gpu_id=0) -> torch.nn.DataParallel:
    print(f'Batch size: {batch_sz}, seq_len: {seq_len}, epochs: {epochs}, gpu_id: {gpu_id}')
    print('Creating dataloader...')
    device = f'cuda:{gpu_id}'
    dataloader, valid_iter, en_vocab, de_vocab, trg_tokenizer, src_tokenizer = create_dataloader(batch_size=batch_sz, seq_len=seq_len, device=device)

    print('Creating model...')
    model = Transformer(seq_len=seq_len, src_vocab_size=len(de_vocab), trg_vocab_size=len(en_vocab))
    model.set_data_parallel(True)
    if pretrained_weghts is not None:
        print('Received a path to pretrained weights: {pretrained_weghts}.')
        model.load_state_dict(torch.load(pretrained_weghts), strict=False)
        print('Successfully loaded pretrained weights.')
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model.to(device)

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
    torch.save(model.module.state_dict(), os.path.join(args.save, WEIGHTS_NAME))
    print('Model is saved.')

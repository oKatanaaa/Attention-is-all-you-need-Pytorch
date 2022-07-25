## Pytorch Transformer

This repository contains an implementation of the original [Attention is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) transformer model (with some minor changes) in PyTorch. 

## Table of Contents
  * [General](#general)
  * [Repo structure](#repo-structure)
  * [Some details](#some-details)
  * [Training experiments](#training-experiments)
  * [Distributed training details](#distributed-training-details)
  * [Inference](#inference)
  * [Training](#training)
  * [Acknowledgements](#acknowledgements)

## General

Implementation wise this is essentially the same as the [gordicaleksa/pytorch-original-transformer](https://github.com/gordicaleksa/pytorch-original-transformer) but with a much simpler training pipeline. I also updated some of the parts of his code base to a newer PyTorch API, so it looks much simpler and works a lot faster.

I have trained my model on Yandex RU-EN and Multi30K EN-DE datasets, but provide weights only for the former. On RU-EN translation I got about 23 BLEU, on EN-DE about 21 BLEU.

The repo contains a single-GPU pipeline as well as distributed training pipeline. 

## Repo structure

Folders:
- `datasets` - you put your dataset here in a separate folder.
- `weights` - weights are saved here.

Files:
- `data_utils.py` - all the data pipeline stuff.
- `datasets.py` - contains a data loader for the Yandex dataset.
- `decoding.py` - contains an implementation of a greedy decoder.
- `model.py` - the actual model code.
- `train.py` - training pipeline.
- `train_distributed.py` - distributed training pipeline.
- `model_distributed.py` - a slight modification of the original model code to make it train on several GPUs.
- `inference.ipynb` - an example of model inference.
- `training.ipynb` - training code, but in a notebook. There is a ready to use pipeline for Multi30K dataset. You can simply run the notebook and observe magic.

## Some details

I don't use sinusoidal positional encoding and stick to the trainable ones (because they are much simpler). Also I replaced ReLU activation function with GELU as it is used in most of the nowadays transformer models.

The Yandex dataset was filtered (the filtering code is available in the YandexDataset class) from Unicode and degenerate pairs of sentences (some pairs are way too different in length, hence there is no good correspondance between them which may hurt models accuracy). Filtering of the data improved BLEU from 18 to 23.

During training I don't do any evaluation since it is very slow (I gave zero ducks to making it fast as I do not intend to get SOTA). I just train the model and see the final result.

I also don't use learning rate schedule from the original paper since in my case this turned out to be suboptimal (the model was training extremely slow). So I just stick to fixed 1e-4 learning rate as it gave me best results.

## Training experiments

| Dataset | Batch size | Sequence length | GPU | Epochs | Time | BLEU |
| - | - | - | - | - | - | - |
| Multi30K EN-DE | 48 | 32 | RTX2060 | 30 | ~60 minutes | 21 |
| Yandex RU-EN | 256 | 64 | NVIDIA V100 | 20 | ~5 hours | 18 |
| Yandex RU-EN filtered | 1024 | 64 | 4x NVIDIA V100 | 30 | ~9 hours | 23 |

As you can see, you can train the model on a middle sized GPU (RTX 2060) in a reasonable amount of time and get interesting results. I saw a guy train his model with GTX 1650 (sequence length 10) for 20 minutes and he also got some decent results. So you don't need a powerful GPU to play with the stuff.

## Distributed training details

I use dummy DataParallel from PyTorch and turns out that it sufficient for this case. Since the tensors are not large (simply contain token indices), there is not much of data transfer overhead and we get sweet over 90% GPU utilization. Though I had to make some tweeks to the model code because the output tensor wouldn't fit in GPUs' memory (which is 30 GB btw!) as it had shape of [batch size, seq len, ~80000] which is stupidly large. I decided to make loss computation a part of the model so that each GPU processes its own part of data and spits out a loss scalar. That does not violate anything related to training dynamics, so don't you worry.

## Trained model

I make my trained model publicly accessible so that you toy with it if you want.

Model weights: https://www.dropbox.com/s/0x94zonw68i066j/yandex_weights.pth?dl=0

RU vocabulary object: https://www.dropbox.com/s/gpehf0244s600cy/ru_vocab.pkl?dl=0

EN vocabulary object: https://www.dropbox.com/s/7qjsuk5utf3xlp1/en_vocab.pkl?dl=0

## Inference

I provide a Jupyter Notebook (inference.ipynb) for model's inference.
Load the weights and vocabs' files into the `weights/Yandex` folder and just run the notebook.
You can insert any text into the decoder and see the result (though it's not formatter). Enjoy.

Example 1:
- input - `Русский язык невероятно сложен.`
- output - `the russian language is incredibly complicated .`
- This is a correct translation!

Example 2:
- input - `Это сложное множество слов, которое трудно перевести.` (ah yes, irony)
- output - `this is a complex and complex set of words that are hard to translate .`
- It is a fairly good translation, but the word `complex` appears twice and that's wrong.
- Speaking of other failure cases, the model doesn't really understand the context. For example, it may translate `предложение` as `offer` whereas (sometimes) it should be `sentence`.

## Training

Preliminarities:
download the [Yandex Dataset](https://translate.yandex.ru/corpus?lang=en) and unarchive it in the `datasets/Yandex` folder. You're good to go.

Single GPU. Run the following:
- `python train_utils.py -bs 48 -e 20 -sl 32 -s weights/{your_folder}`
- Parameters:
  - `-e, --epochs` - number of epochs.
  - `-sl, --seq_len` - sequence length.
  - `-s, --save` - path to the folder where to save the weights.
  - `-pw, --pretrained_weights` - path to the file with pretrained weight to use them as initialization (.pth file).
  - `--gpu_id` - id of the gpu which the model will be trained on.
  - All of the parameters have default values so you might as well just run the script as it is.

Multi GPU. Do the following:
- Set the GPUs' ids in `train_distributed.py`, line 54. By default it's to [0, 1, 2, 3].
- `python train_distributed.py -bs 1024 -e 30 -sl 64 -s weights/{your_folder}`
- Parameters:
  - All of the parameters are the same as in the script above.
  - `--gpu_id` will stand for the 'main gpu' where all the weights are stored.

In case you want simply clone and run stuff, go to `training.ipynb`, there is a ready to use pipeline for Multi30K dataset.


## Acknowledgements

The implementation is inspired by [gordicaleksa/pytorch-original-transformer](https://github.com/gordicaleksa/pytorch-original-transformer). Huge thanks to [Aleksa Gordić](https://github.com/gordicaleksa) for providing his amazing implementation of the transformer model.

```
@misc{Gordić2020PyTorchOriginalTransformer,
  author = {Gordić, Aleksa},
  title = {pytorch-original-transformer},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/gordicaleksa/pytorch-original-transformer}},
}
```
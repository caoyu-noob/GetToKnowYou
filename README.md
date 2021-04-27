# GetToKnowYou
Implementation of paper "Getting To Know You: User Attribute Extraction from Dialogues"

## run
Download "glove.6B.300d.txt" and put it under `glove`.

Download [charNgram.txt]([https://www.logos.t.u-tokyo.ac.jp/~hassy/publications/arxiv2016jmt/jmt_pre-trained_embeddings.tar.gz]) and put it under `CharEmb`.

Put original dataset files under `datasets/triple`.

An example is given in `train_dummy.sh`

The default config will not use char embedding, if you want to use, add `--pretrained_char_emb_file
./CharEmb/charNgram.txt --emb_dim 400 --hidden_dim 400` in `train.sh`.
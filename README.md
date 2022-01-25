# CD-MRC
This repo contains the code of the following paper:

A Consistent Dual-MRC Framework for Emotion-Cause Pair Extraction


## Requirements

- Python 3.6.9
- PyTorch 1.8.0
- transformers 4.11.3




## Quick Start

1. Clone or download this repo.

2. Download the pertrained ["BERT-Base, Chinese"](https://github.com/google-research/bert) model from [this link](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz). And then put the model file `pytorch_model.bin` to the folder `src/bert-base-chinese`.  

3. Run our model:

```bash
python src/main2.py
```

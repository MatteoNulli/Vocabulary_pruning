---
license: apache-2.0
tags:
- generated_from_trainer
datasets:
- multi_news
model-index:
- name: multi_news_longt5_base
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# multi_news_longt5_base

This model is a fine-tuned version of [google/long-t5-tglobal-base](https://huggingface.co/google/long-t5-tglobal-base) on the multi_news dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 8
- eval_batch_size: 1
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3.0

### Framework versions

- Transformers 4.28.1
- Pytorch 2.1.2
- Datasets 2.20.0
- Tokenizers 0.13.3

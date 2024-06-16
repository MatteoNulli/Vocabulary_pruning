---
tags:
- generated_from_trainer
datasets:
- samsum
model-index:
- name: jvelja_t5-large
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# jvelja_t5-large

This model is a fine-tuned version of [jvelja/t5-samsum](https://huggingface.co/jvelja/t5-samsum) on the samsum dataset.

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

- Transformers 4.30.0
- Pytorch 2.3.0
- Datasets 2.19.0
- Tokenizers 0.13.3

#!/bin/bash

pretrain_path=pretrained_model.pth

CUDA_VISIBLE_DEVICES=0 python experiments/measurement_notes/measurement_notes_downstream.py \
  --batch_size 8 \
  --epochs 20 \
  --notes_max_seq_len 256 \
  --measurement_max_seq_len 256 \
  --weight_decay 0.000005 \
  --notes_dropout 0 \
  --measurement_dropout 0 \
  --mlp 128 \
  --measurement_emb_size 128 \
  --notes_emb_size 128 \
  --notes_num_heads 8 \
  --measurement_num_heads 8 \
  --measurement_num_layers 8 \
  --notes_num_layers 8 \
  --measurement_activation GELU \
  --text_model BERT \
  --use_measurements \
  --warmup_epochs 0 \
  --use_pos_emb \
  --lr 0.01 \
  --pretrained_path ${pretrain_path}

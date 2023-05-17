#! /bin/bash
export CUDA_VISIBLE_DEVICES=0

DATA_DIR=one_billion_word_strict_1kw/span_data/processed
MODEL_DIR=checkpoints/exp-gen-span

fairseq-generate $DATA_DIR \
    --path $MODEL_DIR/checkpoint_best.pt \
    --beam 5 \
    --sacrebleu \
    --remove-bpe | tee $MODEL_DIR/generate.sacrebleu.log

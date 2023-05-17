#! /bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
#export MKL_THREADING_LAYER=GNU
# export MKL_SERVICE_FORCE_INTEL=1

DATA_DIR=one_billion_word_strict_1kw/span_data_no_BosEos/processed
MODEL_DIR=checkpoints/exp-gen-span-no-BosEos

python train.py $DATA_DIR \
    --task translation \
    --arch transformer \
    --share-decoder-input-output-embed \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.1 \
    --lr 3e-4 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --max-epoch 20 \
    --dropout 0.3 \
    --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --fp16 \
    --wandb-project exp-gen \
    --save-dir $MODEL_DIR \
    --keep-last-epochs 1 \
    --log-format simple \
    --log-interval 100 \
    --max-tokens 8192 \
    --seed 222 \
    --patience 5 \
    --eval-bleu \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric

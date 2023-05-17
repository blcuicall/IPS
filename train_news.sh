#! /bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
#export MKL_THREADING_LAYER=GNU
# export MKL_SERVICE_FORCE_INTEL=1

DATA_DIR=News/span_data/processed
MODEL_DIR=checkpoints/exp-gen-span-news

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
    --max-epoch 50 \
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
    --patience 8 \
    --eval-bleu \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric

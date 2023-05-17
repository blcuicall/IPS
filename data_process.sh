#! /bin/bash

DATA_DIR=data
for SPLIT in train valid test; do
    for LANG in src tgt; do
        subword-nmt apply-bpe -c $DATA_DIR/raw/subword.code <$DATA_DIR/raw/$SPLIT.$LANG >$DATA_DIR/bpe/$SPLIT.$LANG
    done
done

fairseq-preprocess \
    --task translation \
    --source-lang src \
    --target-lang tgt \
    --trainpref $DATA_DIR/bpe/train \
    --validpref $DATA_DIR/bpe/valid \
    --testpref $DATA_DIR/bpe/test \
    --destdir $DATA_DIR/processed \
    --workers 20

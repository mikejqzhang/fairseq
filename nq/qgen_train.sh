DATA_PATH=/data/mjqzhang/question_generation

TASK=nqgen_full
# TASK=nqgen_date_filtered

TOTAL_NUM_UPDATES=100000
WARMUP_UPDATES=5000
LR=2e-05
MAX_TOKENS=2048
UPDATE_FREQ=4
BART_PATH=$DATA_PATH/bart/bart.large/model.pt

MKL_SERVICE_FORCE_INTEL=1 CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train $DATA_PATH/$TASK-bin \
    --save-dir $DATA_PATH/saved_models/$TASK \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --task translation \
    --source-lang src --target-lang tgt \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_large \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters;

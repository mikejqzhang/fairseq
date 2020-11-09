DATA_PATH=/data/mjqzhang/question_generation

# TASK=nqgen_block
TASK=nqgen_sent

for SPLIT in train dev
do
  for LANG in src tgt
  do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json $DATA_PATH/bart/encoder.json \
    --vocab-bpe $DATA_PATH/bart/vocab.bpe \
    --inputs "$DATA_PATH/$TASK/$SPLIT.$LANG" \
    --outputs "$DATA_PATH/$TASK/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done

fairseq-preprocess \
  --source-lang "src" \
  --target-lang "tgt" \
  --trainpref "$DATA_PATH/${TASK}/train.bpe" \
  --validpref "$DATA_PATH/${TASK}/dev.bpe" \
  --destdir "$DATA_PATH/${TASK}-bin/" \
  --workers 60 \
  --srcdict $DATA_PATH/bart/dict.txt \
  --tgtdict $DATA_PATH/bart/dict.txt

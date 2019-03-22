export BERT_BASE_DIR=./uncased_L-12_H-768_A-12

export DATASET=./data/

CUDA_VISIBLE_DEVICES=3 python run_template.py \
  --data_dir=$DATASET \
  --task_name=template \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --output_dir=./output_template \
  --do_train=false \
  --do_eval=true \
  --do_predict=false \
  --max_seq_length=32 \
  --train_batch_size=32 \
  --learning_rate=5e-5\
  --num_train_epochs=50.0

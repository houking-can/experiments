export BERT_BASE_DIR=./uncased_L-12_H-768_A-12

export DATASET=./data/

CUDA_VISIBLE_DEVICES=3 python run_classifier.py \
  --data_dir=$DATASET \
  --task_name=abstract \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --output_dir=./output_abstract/ \
  --do_predict=true \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=5e-5\
  --num_train_epochs=100.0

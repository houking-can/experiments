CUDA_VISIBLE_DEVICES=3  python /home/yhj/long-summarization/run_summarization.py \
--mode=decode \
--data_path=/home/yhj/long-summarization/data/bin_data/test.bin \
--vocab_path=/home/yhj/long-summarization/data/vocab \
--log_root=logroot \
--exp_name=baseline-introduction \
--max_dec_steps=210 \
--max_enc_steps=1200 \
--num_sections=2 \
--max_section_len=1000 \
--batch_size=4 \
--vocab_size=50000 \
--use_do=True \
--optimizer=adagrad \
--do_prob=0.25 \
--hier=True \
--split_intro=True \
--fixed_attn=True \
--legacy_encoder=False \
--coverage=False \
--single_pass=true

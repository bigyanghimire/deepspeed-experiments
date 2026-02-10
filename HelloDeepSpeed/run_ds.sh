deepspeed --num_gpus=2 --master_port=29600 train_bert_ds.py --checkpoint_dir experiment_deepspeed $@

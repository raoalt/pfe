#!/bin/bash


nvidia-smi
python train.py -task ext -mode test -bert_data_path ../bert_data/cnndm -ext_dropout 0.1 -model_path ../models/ext_cnndm_sbert -lr 2e-3 -visible_gpus -1 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -train_steps 50000 -accum_count 2 -log_file ../logs/test_ext_cnndm_sbert -use_interval true -warmup_steps 10000 -max_pos 512 -temp_dir ../temp_ext_cnndm -test_from ../models/ext_cnndm_sbert/model_step_44000.pt

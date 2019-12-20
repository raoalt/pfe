#!/bin/bash

nvidia-smi
python train.py -task ext -mode validate -batch_size 3000 -test_batch_size 500 -bert_data_path ../bert_data/cnndm -log_file ../logs/val_ext_sbert_cnndm -model_path  ../models/ext_cnndm_sbert -sep_optim true -use_interval true -visible_gpus 0,1,2 -max_pos 512 -min_length 20 -max_length 100 -alpha 0.9 -result_path ../logs/validate_ext_sbert_cnndm -test_all

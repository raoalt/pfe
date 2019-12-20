#!/bin/bash 
#NGPUS=3
GPU_IDS=0,1,2,3
CUDA_VISIBLE_DEVICES=${GPU_ID} 
#SBATCH --ngres=gpu:3
#SBATCH --cpus-per-task=12
nvidia-smi
python train.py -task ext -mode train -bert_data_path ../bert_data/xsum -ext_dropout 0.1 -model_path ../models/ext_xsum_sbert -lr 2e-3 -visible_gpus 0,1,2 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -train_steps 50000 -accum_count 2 -log_file ../logs/ext_xsum_sbert -use_interval true -warmup_steps 10000 -max_pos 512 -temp_dir ../temp_ext_xsum
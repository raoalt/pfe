#!/bin/bash
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=14

srun test_ext_cnndm.sh
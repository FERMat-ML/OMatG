#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=13
##SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=30GB
#SBATCH --job-name=gpu_hyper_training
#SBATCH --output=train.out

# Initialize
module purge
source ~/.bashrc
module load cuda/11.6.2
cd /scratch/tje3676/RESEARCH/FERMAT/STAGE1/OMG

# Run
singularity exec --nv\
	    --overlay /scratch/tje3676/RESEARCH/FERMat-env.ext3:ro \
	    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif\
	    /bin/bash -c "source /ext3/env.sh; python scripts/main.py predict --config omg/conf_examples/test_config_ode.yaml"

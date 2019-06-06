#!/bin/bash
#SBATCH -p high
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -C intel
#SBATCH --array=1-20:1
#SBATCH --workdir=/homedtic/mwon/codes/music-tagging-attention/preprocessing/msd/
#SBATCH -o logs/iter_%a.out

 
module load Python/3.6.4-foss-2017a 

#source /homedtic/mwon/venv_amd/bin/activate
source /homedtic/mwon/envs/intel/bin/activate

python -u preprocess.py run '/datasets/MTG/audio/incoming/millionsong-audio/mp3/' '/homedtic/mwon/dataset/msd/' ${SLURM_ARRAY_TASK_ID} 20

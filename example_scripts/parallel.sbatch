#!/bin/bash

# email on start, end, and abortion
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sivadasetty@rcc.uchicago.edu

#SBATCH --job-name=integrin

#SBATCH -o slurm_integrin_string.o%j    # Name of stdout output file
#SBATCH -e slurm_integrin_string.e%j    # Name of stderr error file
#SBATCH --partition=fela-cpu #gm4-pmext #andrewferguson-gpu
##SBATCH --qos=gm4
#SBATCH --account=pi-andrewferguson
#SBATCH --nodes=1            # SET NUM NODES
##SBATCH --gres=gpu:1        # SET NUM GPUS
#SBATCH --ntasks-per-node=1  # SETS NUM MPI RANKS (1 PER GPU)
#SBATCH --cpus-per-task=20    # SET NUM THREADS

#SBATCH --time=60:00:00

module unload cuda python
module load python
module load cuda/11.2

source activate /project2/andrewferguson/sivadasetty/doe/analysis-integrin/string_mechanisms/seek

source /project2/andrewferguson/sivadasetty/doe/analysis-integrin/string_mechanisms/start_seek.sh

module unload python

which python

source format.sh
python integrin_string.py &> out_parallel.out



#!/bin/bash


# email on start, end, and abortion
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sivadasetty@rcc.uchicago.edu

#SBATCH --job-name=string_integrin
#SBATCH -o logs/array_%A_%a.o%j
#SBATCH -e logs/array_%A_%a.e%j

#SBATCH --array=0-5
#SBATCH --partition=fela #gpu2 #gm4-pmext #fela #andrewferguson-gpu
##SBATCH --qos=gm4
#SBATCH --account=pi-andrewferguson
#SBATCH --nodes=1            # SET NUM NODES
#SBATCH --gres=gpu:1         # SET NUM GPUS

#SBATCH --time=30:00:00

##SBATCH --exclusive
##SBATCH --mem=0

##SBATCH --ntasks-per-node=20  # SETS NUM MPI RANKS (1 PER GPU)
#SBATCH --mem-per-cpu=4000
#SBATCH --cpus-per-task=10    # SET NUM THREADS


module load cuda/11.2
source activate /home/sivadasetty/scratch-midway2/pdbfixer

#ulimit -l unlimited

module unload cuda python
module load python
module load cuda/11.2

source activate /project2/andrewferguson/sivadasetty/doe/analysis-integrin/string_mechanisms/seek

source /project2/andrewferguson/sivadasetty/doe/analysis-integrin/string_mechanisms/start_seek.sh

module unload python


which python


args=("$@")
echo Number of arguments: $#
echo 1st argument: ${args[0]}
echo 2nd argument: ${args[1]}
echo 3rd argument: ${args[2]}
#for i in {0..9}
#do
#echo $i
mkdir slurm_logs

#python run_each_image_init_iter_singleGPU.py $SLURM_ARRAY_TASK_ID ${args[0]} ${args[1]} $SLURM_ARRAY_TASK_ID > slurm_logs/iter"${args[2]}"_out_"$SLURM_ARRAY_TASK_ID".out

python run_each_image_singleGPU.py $SLURM_ARRAY_TASK_ID ${args[0]} ${args[1]} $SLURM_ARRAY_TASK_ID > slurm_logs/iter"${args[2]}"_out_"$SLURM_ARRAY_TASK_ID".out


#done





#!/bin/bash  
#SBATCH --job-name=JOBNAME 
#SBATCH --partition=gpu           
#SBATCH --nodes=1                 
#SBATCH -n 64    
#SBATCH --gres=gpu:a100:1
#SBATCH -A PDS0352
#SBATCH --output=/users/PDS0352/wyang107/project/LCEG/longbench_pro/result/output.txt      
#SBATCH --error=/users/PDS0352/wyang107/project/LCEG/longbench_pro/result/error.txt   
#SBATCH --time=0-10:00:00

singularity exec --nv /users/PDS0352/wyang107/images/pytorch.2.4.1-cuda12.1-cudnn9-devel.sif bash -c  "python /users/PDS0352/wyang107/project/LCEG/longbench_pro/pred.py"


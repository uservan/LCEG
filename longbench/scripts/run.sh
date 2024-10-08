#!/bin/bash  
#SBATCH --job-name=JOBNAME 
#SBATCH --partition=gpu           
#SBATCH --nodes=1                 
#SBATCH -n 64    
#SBATCH --gres=gpu:h100:1
#SBATCH -A PDS0352
#SBATCH --output=/users/PDS0352/wyang107/project/LCEG/longbench/result/output.txt      
#SBATCH --error=/users/PDS0352/wyang107/project/LCEG/longbench/result/error.txt   
#SBATCH --time=0-10:00:00

python /users/PDS0352/wyang107/project/LCEG/longbench/pred.py --model llama2-7b-hf

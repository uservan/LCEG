#!/bin/bash  
#SBATCH --job-name=JOBNAME 
#SBATCH --partition=gpu           
#SBATCH --nodes=1                 
#SBATCH -n 64    
#SBATCH --gres=gpu:h100:1
#SBATCH -A PDS0352
#SBATCH --output=/users/PDS0352/wyang107/project/LCEG/longbench_pro/result/output4.txt      
#SBATCH --error=/users/PDS0352/wyang107/project/LCEG/longbench_pro/result/error4.txt   
#SBATCH --time=0-10:00:00

python /users/PDS0352/wyang107/project/LCEG/longbench_pro/pred.py --model llama-3.1-8B-Instruct

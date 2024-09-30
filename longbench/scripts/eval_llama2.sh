#!/bin/bash  
#SBATCH --job-name=JOBNAME 
#SBATCH --partition=gpu           
#SBATCH --nodes=1                 
#SBATCH -n 64     
#SBATCH --gres=gpu:a100:1
#SBATCH -A PDS0352
#SBATCH --output=/users/PDS0352/wyang107/project/LCEG/longbench/result/output.txt      
#SBATCH --error=/users/PDS0352/wyang107/project/LCEG/longbench/result/error.txt   
#SBATCH --time=0-10:00:00


### longbench 
datasets=("qasper" "multifieldqa_en" "hotpotqa" "2wikimqa" "musique" \
          "gov_report" "qmsum" "multi_news" "trec" "triviaqa" "samsum" \
          "passage_count" "passage_retrieval_en" "lcc" "repobench-p" "narrativeqa")

### many shots trec
# datasets=("trec_1000shots" "trec_875shots" "trec_750shots" "trec_625shots" "trec_500shots" "trec_400shots" "trec_300shots" "trec_200shots" "trec_100shots" "trec_75shots" "trec_50shots" "trec_25shots" "trec_10shots" "trec_5shots" "trec_1shots")


### longbench-e
# datasets=( "qasper" "multifieldqa_en" "hotpotqa" "2wikimqa" \
#           "gov_report" "multi_news" "trec" "triviaqa" "samsum" \
#           "passage_count" "passage_retrieval_en" "lcc" "repobench-p")

### models
models=("llama2-7b-hf-slimpajama-yarn-32k")
# models=(
#     "llama2-7b-hf" \
#     "llama2-7b-hf-slimpajama-landmark" \
#     "llama2-7b-hf-lminfinite" \
#     "llama2-7b-hf-slimpajama-pi-32k" \
#     "llama2-7b-hf-slimpajama-longlora-32k" \ 
#     "llama2-7b-hf-ntk-frozen" \
#     "llama2-7b-hf-slimpajama-ntk-32k" \
#     "llama2-7b-hf-slimpajama-ntk-64k" \
#     "llama-2-7b-hf-slimpajama-ntk-64k-2B" \
#     "llama2-7b-hf-slimpajama-yarn-32k"
#     )

### models test 4k
# models=(
#     "llama2-7b-hf-slimpajama-landmark-test4k" \
#     "llama2-7b-hf-lminfinite-test4k" \
#     "llama2-7b-hf-slimpajama-pi-32k-test4k" \
#     "llama2-7b-hf-slimpajama-longlora-32k-test4k"  \
#     "llama2-7b-hf-ntk-frozen-test4k" \
#     "llama2-7b-hf-slimpajama-ntk-64k-test4k" \
#     "llama-2-7b-hf-slimpajama-ntk-64k-2B-test4k" \
#     "llama2-7b-hf-slimpajama-yarn-32k-test4k"
# )

for dataset in "${datasets[@]}";
do
for MODEL_NAME in  "${models[@]}"; 
do
echo "$dataset"
/workspace/LCEG/longbench/pred.py \
--model ${MODEL_NAME} \
--dataset_name ${dataset} 
done
done


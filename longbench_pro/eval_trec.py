import os
import json
import argparse
import numpy as np
from tqdm import tqdm


def set_global_path(path):
    return os.path.join('/users/PDS0352/wyang107/project/LCEG/longbench_pro', path)

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
    acc_score
)

dataset2metric = {
    'qa':qa_f1_score,
    'sum':rouge_score,
    "passage_count": acc_score,
    "passage_retrieval": acc_score,
    "counting_stars": acc_score,
    "kv_retrieval": acc_score,

    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    # "passage_retrieval_en": retrieval_score,
    # "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='llama-3.1-8B-Instruct')
    return parser.parse_args(args)

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    datasets_type = ["qa",'sum','passage_count','passage_retrieval', 'counting_stars', 'kv_retrieval']
    for d in datasets_type:
        if d in dataset: dataset=d
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ['qa','passage_count','passage_retrieval','counting_stars','kv_retrieval']:
            prediction = prediction.lstrip('\n').split('\n')[0]
        # if dataset in []:
        #     match = re.search(r'\d+', prediction)
        #     if match: prediction = match.group()
        #     else: prediction=''
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)

if __name__ == '__main__':
    args = parse_args()
    scores = dict()
    path = set_global_path(f"pred/{args.model}/")
    all_files = sorted(os.listdir(path))
    print("Evaluating on:", all_files)
    for filename in tqdm(all_files):
        if not filename.endswith("jsonl"):
            continue
        predictions, answers, lengths = [], [], []
        dataset = filename.split('.')[0]
        with open(f"{path}{filename}", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                if type(data["answers"]) != list: 
                    data["answers"] = [data["answers"]]
                answers.append(data["answers"])
                all_classes = filename
                if "length" in data:
                    lengths.append(data["length"])
        score = scorer(dataset, predictions, answers, all_classes)
        scores[dataset] = score
    out_path = set_global_path(f"pred/{args.model}/result.json")
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

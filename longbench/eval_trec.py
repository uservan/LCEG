import os
import json
import argparse
import numpy as np
import math

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
)

dataset2metric = {
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
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

def set_global_path(path):
    return os.path.join('/users/PDS0352/wyang107/project/LCEG/longbench', path)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='llama2-7b-hf-slimpajama-yarn-32k')
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E", default=True)
    return parser.parse_args(args)

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-2k": [],"2-4k": [], "4-6k": [], "6-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in [ # ["qasper", "trec", "triviaqa", "samsum", "lsht"]
                "narrativeqa", "qasper", "multifieldqa_en",
                "hotpotqa", "2wikimqa", "musique",
                "trec", "triviaqa", "samsum", "lsht",
                "passage_count", "passage_retrieval_en"
                ] or "trec" in dataset:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 2000:
            scores["0-2k"].append(score)
        elif length < 4000:
            scores["2-4k"].append(score)
        elif length < 6000:
            scores["4-6k"].append(score)
        elif length < 8000:
            scores["6-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        # s = round(100 * np.mean(scores[key]), 2)
        # if math.isnan(s): s= 0
        if len(scores[key]) == 0: s=0
        else: s = round(100 * sum(scores[key])/len(scores[key]), 2)
        scores[key] = {'score': s, 'num':len(scores[key])}
    return scores

def scorer(dataset, predictions, answers, all_classes):
    total_score = []
    if "trec" in dataset:
        dataset="trec"
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        post_process_list = [
                "narrativeqa", "qasper", "multifieldqa_en",
                "hotpotqa", "2wikimqa", "musique",
                "trec", "triviaqa", "samsum", "lsht",
                "passage_count", "passage_retrieval_en"
                ]
        if dataset in post_process_list:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score.append(score)
    return round(100 * sum(total_score) / len(total_score), 2)

if __name__ == '__main__':
    args = parse_args()
    scores = dict()
    # if args.e:
    #     path = set_global_path(f"pred_e/{args.model}/")
    # else:
    #    path = set_global_path(f"pred/{args.model}/")
    path = set_global_path(f"pred/{args.model}/")
    all_files = os.listdir(path)
    print("Evaluating on:", all_files)
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        predictions, answers, lengths = [], [], []
        dataset = filename.split('.')[0]
        with open(f"{path}{filename}", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
                if "length" in data:
                    lengths.append(data["length"])
        if args.e:
            score = scorer_e(dataset, predictions, answers, lengths, all_classes)
        else:
            # score = scorer_e(dataset, predictions, answers, lengths, all_classes)
            score = scorer(dataset, predictions, answers, all_classes)
        scores[dataset] = score
    if args.e:
        out_path = set_global_path(f"pred/{args.model}/result.json")
    else:
        out_path = set_global_path(f"pred/{args.model}/result_all.json")
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

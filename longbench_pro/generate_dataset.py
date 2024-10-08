import os
from datasets import load_dataset
import datasets
from datasets import Value, Sequence
import sys
import math
import transformers
# from openai import OpenAI
import random
import re
import json
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.dirname(current_dir)
sys.path.append(models_dir)


# client = OpenAI(
#   organization='org-phGCBrHmAU0qDIFg26G1UnSQ',
#   project='proj_TvuTyHB0YqppPdFvxdrkuAUk',
# )
# response = client.chat.completions.create(
#     model='gpt-3.5-turbo', # "gpt-4o-mini",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Hello! Can you explain quantum computing?"}
#     ],
# )
# print(response.choices[0].message.content)


cache_dir = '/users/PDS0352/wyang107/project/LCEG/model_cache/data'

# 1.load dataset: {"input": obj['input'], 'context':obj['context'] , 'answers':obj['answers'], 'source':obj['dataset'], "length": obj["length"]}
# 2. 生成数据
#   single-doc-qa
#   multi-doc-qa
#   sum
#   Synthetic

# code diag

def get_dataset(path='LongBench', name='narrativeqa'):
    new_dataset = list()
    if path == 'LongBench':
        dataset = load_dataset('THUDM/LongBench',name=name, split='test', cache_dir=cache_dir, streaming= True)
        for obj in dataset:
            new_dataset.append({"input": [obj['input']], "length": len(tokenizer.encode(obj['context'])), 
                            'context':obj['context'] , 'answers':[obj['answers']],
                            'dataset':name})
    if path == 'LEval':
        dataset = load_dataset('L4NLP/LEval', name=name  ,split='test', cache_dir=cache_dir, streaming= True)
        for obj in dataset:
            new_dataset.append({"input": obj['instructions'], "length": len(tokenizer.encode(obj['input'])), 
                            'context':obj['input'] , 'answers':obj['outputs'],
                            'dataset':name})
    return new_dataset

tokenizer = transformers.AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')

def generate_dataset_single_doc_qa(length=8, rows=100):
    # single-doc-qa
    length = length *(2**10)
    # load raw datasets
    dataset_list, new_datasets, choices=list(), [], []
    for name in ("qasper", "multifieldqa_en", "narrativeqa"):
        dataset = get_dataset('LongBench',name)
        dataset_list.extend(dataset)
    for name in ('multidoc_qa', 'legal_contract_qa', 'financial_qa', 'natural_question', 'scientific_qa' ):
        dataset = get_dataset('LEval',name)
        dataset_list.extend(dataset)
    # generate
    datasets_id = [(data['length'], i) for i, data in enumerate(dataset_list)]
    for i in range(rows):
        l = length
        while l > 0:
            datasets_id_tmp = [data_id for data_id in datasets_id if data_id[0]<l]
            if len(datasets_id_tmp)<=0: break
            choice = random.choice(datasets_id_tmp)
            choices.append(choice)
            l = l-choice[0]
        new_datasets.append(choices)
        choices = []
    # collect
    results = []
    for choices in new_datasets:
        context = ''
        for i, c in enumerate(choices):
            context = context + f'Passage {i+1}:\n' + dataset_list[c[1]]['context'] + '\n\n'
        c_i = random.choice(range(len(choices)))
        d = dataset_list[choices[c_i][1]]
        results.append({ "length": len(tokenizer.encode(context)),  "new_context": context, 'old_context': d['context'] , 
                        'instruction': f'Answer the question related with Passage {c_i+1}. ', 
                         "input": d['input'], "answers": d["answers"]})
    return results

def generate_dataset_multi_doc_qa(length=8, rows=100):
    # multi-doc-qa
    length = length *(2**10)
    # load noise datasets
    noise_dataset_list = []
    for name in ("qasper", "multifieldqa_en", "narrativeqa"):
        dataset = get_dataset('LongBench',name)
        noise_dataset_list.extend(dataset)
    for name in ('multidoc_qa', 'legal_contract_qa', 'financial_qa', 'natural_question', 'scientific_qa' ):
        dataset = get_dataset('LEval',name)
        noise_dataset_list.extend(dataset)
    # load multi-doc datasets
    dataset_list, new_datasets = [], []
    for name in ("hotpotqa", "2wikimqa", "musique"):
        dataset = get_dataset('LongBench',name)
        dataset_list.extend(dataset)
    # generate
    datasets_id, noise_datasets_id ,choices = [(data['length'], i) for i, data in enumerate(dataset_list)], \
        [(data['length'], i) for i, data in enumerate(noise_dataset_list)],[]
    for i in range(rows):
        l = length
        datasets_id_tmp = [data_id for data_id in datasets_id if data_id[0]<l]
        choice = random.choice(datasets_id_tmp)
        choices.append(choice) 
        l = l-choice[0]     
        while l > 0:
            datasets_id_tmp = [data_id for data_id in noise_datasets_id if data_id[0]<l]
            if len(datasets_id_tmp)<=0: break
            choice = random.choice(datasets_id_tmp)
            choices.append(choice)
            l = l-choice[0]
        new_datasets.append(choices)
        choices = []
    # collect
    results = []
    for choices in new_datasets:
        qa = dataset_list[choices[0][1]]
        split_text = [ (p, True) for p in re.split(r'Passage \d+:\n', qa['context']) if p.strip()]
        noise_text = [(noise_dataset_list[c[1]]['context'] ,False) for c in choices[1:]]
        split_text.extend(noise_text)
        random.shuffle(split_text)
        context, num = '', []
        for i, (text, flag) in enumerate(split_text):
            context = context + f'Passage {i+1}:\n' + text + '\n'
            if flag: num.append(i+1)
        results.append({'instruction':f'Answer the question related with Passage '+','.join(map(str, num))+'. ' , 
                        "input": qa['input'], "answers": qa["answers"], "new_context": context,  "old_context": qa['context'],
                        "length": len(tokenizer.encode(context))})
    return results

def generate_dataset_single_doc_sum(length=8, rows=100):
    # single-doc-sum
    length = length *(2**10)
    # load raw datasets
    dataset_list, new_datasets, choices=list(), [], []
    for name in ("gov_report", "qmsum"): # "multi_news"
        dataset = get_dataset('LongBench',name)
        dataset_list.extend(dataset)
    for name in ( 'patent_summ','tv_show_summ','review_summ',  'meeting_summ' ): # 'news_summ',
        dataset = get_dataset('LEval',name)
        dataset_list.extend(dataset)
    # generate
    datasets_id = [(data['length'], i) for i, data in enumerate(dataset_list)]
    for i in range(rows):
        l = length
        while l > 0:
            datasets_id_tmp = [data_id for data_id in datasets_id if data_id[0]<l]
            if len(datasets_id_tmp)<=0: break
            choice = random.choice(datasets_id_tmp)
            choices.append(choice)
            l = l-choice[0]
        new_datasets.append(choices)
        choices = []
    # collect
    results = []
    for choices in new_datasets:
        context = ''
        for i, c in enumerate(choices):
            context = context + f'Passage {i+1}:\n' + dataset_list[c[1]]['context'] + '\n\n'
        c_i = random.choice(range(len(choices)))
        d = dataset_list[choices[c_i][1]]
        instruction = f'Write a one-page summary of Passage {c_i+1} into a few short sentences'
        results.append({ 'instruction': '',
                         "input": [f'{instruction}: {inp}' if len(inp) > 0 else f'{instruction}.' for inp in d['input']], 
                         "answers": d["answers"], "new_context": context, 'old_context':d['context'],
                         "length": len(tokenizer.encode(context))})
    return results

def generate_dataset_multi_doc_sum(length=8, rows=100):
    # multi-doc-sum
    length = length *(2**10)
    # load noise datasets
    noise_dataset_list = []
    for name in ("gov_report", "qmsum"):
        dataset = get_dataset('LongBench',name)
        noise_dataset_list.extend(dataset)
    for name in ('patent_summ','tv_show_summ','review_summ',  'meeting_summ' ):
        dataset = get_dataset('LEval',name)
        noise_dataset_list.extend(dataset)
    # load multi-doc datasets
    dataset_list, new_datasets = [], []
    for name in ("multi_news_e", ):
        dataset = get_dataset('LongBench',name)
        dataset_list.extend(dataset)
    # generate
    datasets_id, noise_datasets_id ,choices = [(data['length'], i) for i, data in enumerate(dataset_list)], \
        [(data['length'], i) for i, data in enumerate(noise_dataset_list)],[]
    for i in range(rows):
        l = length
        datasets_id_tmp = [data_id for data_id in datasets_id if data_id[0]<l]
        choice = random.choice(datasets_id_tmp)
        choices.append(choice) 
        l = l-choice[0]     
        while l > 0:
            datasets_id_tmp = [data_id for data_id in noise_datasets_id if data_id[0]<l]
            if len(datasets_id_tmp)<=0: break
            choice = random.choice(datasets_id_tmp)
            choices.append(choice)
            l = l-choice[0]
        new_datasets.append(choices)
        choices = []
    # collect
    results = []
    for choices in new_datasets:
        qa = dataset_list[choices[0][1]]
        split_text = [ (p, True) for p in re.split(r'Passage \d+:\n', qa['context']) if p.strip()]
        noise_text = [(noise_dataset_list[c[1]]['context'] ,False) for c in choices[1:]]
        split_text.extend(noise_text)
        random.shuffle(split_text)
        context, num = '', []
        for i, (text, flag) in enumerate(split_text):
            context = context + f'Passage {i+1}:\n' + text + '\n'
            if flag: num.append(f'Passage {i+1}')
        instruction = 'Write a one-page summary of selected passages only including '+','.join(map(str, num))
        results.append({'instruction':'' , 
                        "input": qa['input'], 
                        "input": [f'{instruction}: {inp}' if len(inp) > 0 else f'{instruction}.' for inp in qa['input']], 
                        "answers": qa["answers"], "new_context": context,  "old_context": qa['context'],
                        "length": len(tokenizer.encode(context))})
    return results

# // "multi_doc_sum":"You are given several passages as follows and these passages are from many different fields. {input}\n\n{context}\n\nNow, {input}\n\nSummary:",
# "single_doc_sum":"You are given several passages as follows and these passages are from many different fields. {input}\n\n{context}\n\nNow, {input}\n\nSummary:",
    

func = {
        # 'single_doc_qa':generate_dataset_single_doc_qa,
        # 'multi_doc_qa': generate_dataset_multi_doc_qa,
        'single_doc_sum':generate_dataset_single_doc_sum,
        'multi_doc_sum':generate_dataset_multi_doc_sum
}
rows = 200
out_path = '/users/PDS0352/wyang107/project/LCEG/longbench_pro/data'
for length in tqdm([8, 16, 31]):
    for key in func.keys():
        result = func[key](length, rows)
        with open(os.path.join(out_path, f'{key}_{length}.jsonl'), "w", encoding="utf-8") as f:
            for pred in result:
                json.dump(pred, f, ensure_ascii=False)
                f.write('\n')



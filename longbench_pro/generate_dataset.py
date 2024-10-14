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
import uuid

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


# 1.load dataset: {"input": obj['input'], 'context':obj['context'] , 'answers':obj['answers'], 'source':obj['dataset'], "length": obj["length"]}
# 2. 生成数据
#   single-doc-qa
#   multi-doc-qa
#   sum
#   Synthetic

# code diag
cache_dir = '/users/PDS0352/wyang107/project/LCEG/model_cache/data'
token='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'
tokenizer = transformers.AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', token=token)

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

def generate_dataset_kv_retrieval(length=8, rows=100, kv_num=2):
    length = length *(2**10)
    # load raw datasets
    dataset_list, new_datasets=[], []
    for name in ("qasper", "multifieldqa_en", "narrativeqa"):
        dataset = get_dataset('LongBench',name)
        dataset_list.extend(dataset)
    for name in ('multidoc_qa', 'legal_contract_qa', 'financial_qa', 'natural_question', 'scientific_qa' ):
        dataset = get_dataset('LEval',name)
        dataset_list.extend(dataset)
    # generate
    datasets_id = [(data['length'], i) for i, data in enumerate(dataset_list)]
    while len(new_datasets) < rows:
        choices = []
        l = length
        while l > 0:
            datasets_id_tmp = [data_id for data_id in datasets_id if data_id[0]<l]
            if len(datasets_id_tmp)<=0: break
            choice = random.choice(datasets_id_tmp)
            choices.append(choice)
            l = l-choice[0]
        if len(choices)<kv_num: continue
        new_datasets.append(choices)
    # collect
    results = []
    for choices in new_datasets:
        context, keys, position = '', [str(uuid.uuid4()) for _ in range(kv_num+2)], random.sample(range(len(choices)), kv_num)
        key_value = dict()
        for i, p in enumerate(position):
            key_value[p] = f'The pass value of {keys[i]} is {keys[i+1]}.'
        for i, c in enumerate(choices):
            context = context + f'Passage {i+1}:\n' + dataset_list[c[1]]['context']
            if i in key_value.keys():
                context = context + key_value[i]
            context = context + '\n\n'
        question_key = keys[:4]
        random.shuffle(question_key)
        results.append({ "length": len(tokenizer.encode(context)),  "new_context": context, 'old_context': '' , 
                        'instruction': '', 
                         "input": [f'What is the pass value of the pass value of {keys[0]} ?'+
                                   f'\n1. {question_key[0]}\n2. {question_key[1]}\n3. {question_key[2]}\n4. {question_key[3]}\n'+
                                   'Please provide your answer as a single number (1, 2, 3, or 4) without any explanation.']
                                   ,"answers":[[question_key.index(keys[2])+1,keys[2]]]})
    return results

def generate_dataset_counting_stars(length=8, rows=100, test_type='Acquisition'):
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
    while len(new_datasets) < rows:
        choices = []
        l = length
        while l > 0:
            datasets_id_tmp = [data_id for data_id in datasets_id if data_id[0]<l]
            if len(datasets_id_tmp)<=0: break
            choice = random.choice(datasets_id_tmp)
            choices.append(choice)
            l = l-choice[0]
        if len(choices)<4: continue
        new_datasets.append(choices)
    # collect
    results = []
    def exchange(my_list):
        index1, index2 = random.sample(range(len(my_list)), 2)
        my_list[index1], my_list[index2] = my_list[index2], my_list[index1]
        return my_list
    for choices in new_datasets:
        context, whole_stars = '', 0
        answers = []
        for i, c in enumerate(choices):
            context = context + '\n' + dataset_list[c[1]]['context']
            a_stars, r_stars = random.randint(1, 100), random.randint(1, 100)
            if test_type == 'Acquisition':
                single_star = f"\nThe little penguin counted {a_stars} ★\n"
            if test_type == 'Reasoning':
                single_star = f"\nThe little penguin counted {r_stars} ★, but found that a mistake had been made, so the counting was done again, and this time {a_stars} ★ was counted correctly.\n"
            whole_stars = whole_stars+a_stars
            context = context+ single_star
            answers.append(a_stars)
        if test_type == 'Acquisition':
            question = f"On this moonlit and misty night, the little penguin is looking up at the sky and concentrating on counting ★. Please help the little penguin collect the number of ★, for example: [x, x, x,...]. The summation is not required, and the numbers in [x, x, x,...]. represent the counted number of ★ by the little penguin. "
        if test_type == 'Reasoning':
            question = f"On this moonlit and misty night, the little penguin is looking up at the sky and concentrating on counting ★. Please help the little penguin collect the correct number of ★, for example: [x, x, x,...]. The summation is not required, and the numbers in [x, x, x,...]. represent the correctly counted number of ★ by the little penguin. "
        question_key = [exchange(answers[:]) for i in range(3)]+[answers[:]]
        random.shuffle(question_key)
        question=question+f'\n1. {question_key[0]}\n2. {question_key[1]}\n3. {question_key[2]}\n4. {question_key[3]}\nPlease provide your answer as a single number (1, 2, 3, or 4) without any explanation.'
        results.append({ "length": len(tokenizer.encode(context)),  "new_context": context, 'old_context': '' , 
                        'instruction': '',  "input": [question], "answers": [[question_key.index(answers)+1, f'{answers}']]})
    return results

def generate_dataset_passage_count(length=8, rows=100):
    length = length *(2**10)
    # load raw datasets
    dataset_list, new_datasets=[], []
    for name in ("qasper", "multifieldqa_en", "narrativeqa"):
        dataset = get_dataset('LongBench',name)
        dataset_list.extend(dataset)
    for name in ('multidoc_qa', 'legal_contract_qa', 'financial_qa', 'natural_question', 'scientific_qa' ):
        dataset = get_dataset('LEval',name)
        dataset_list.extend(dataset)
    # generate
    datasets_id = [(data['length'], i) for i, data in enumerate(dataset_list)]
    while len(new_datasets) < rows:
        choices, l = [], length
        while l > 0:
            passage_num = random.sample(range(3), 1)[0]+1
            datasets_id_tmp = [data_id for data_id in datasets_id if data_id[0]*passage_num<l]
            if len(datasets_id_tmp)<=0: break
            choice = random.choice(datasets_id_tmp)
            choices.append((choice, passage_num))
            l = l-choice[0]
        new_datasets.append(choices)
    # collect
    results = []
    for choices in new_datasets:
        passages_num = len(choices)
        choices = [c for (c, num) in choices for i in range(num)]
        random.shuffle(choices)
        context = ''
        for i, c in enumerate(choices):
            context = context + f'Passage {i+1}:\n' + dataset_list[c[1]]['context'] + '\n\n'
        results.append({ "length": len(tokenizer.encode(context)),  "new_context": context, 'old_context': '' , 
                        'instruction': '',  "input":[''], "answers": [f'{passages_num}']})
    return results

def generate_dataset_passage_retrieval(length=8, rows=100):
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
        results.append({ 'instruction': '',
                         "input": [d["answers"][0]], "answers": [f'{c_i+1}'] , "new_context": context, 
                         'old_context':d['context'], "length": len(tokenizer.encode(context))})
    return results


# // "multi_doc_sum":"You are given several passages as follows and these passages are from many different fields. {input}\n\n{context}\n\nNow, {input}\n\nSummary:",
# "single_doc_sum":"You are given several passages as follows and these passages are from many different fields. {input}\n\n{context}\n\nNow, {input}\n\nSummary:",
    

func = {
        'single_doc_qa':generate_dataset_single_doc_qa,
        'multi_doc_qa': generate_dataset_multi_doc_qa,
        'single_doc_sum':generate_dataset_single_doc_sum,
        'multi_doc_sum':generate_dataset_multi_doc_sum,
        'kv_retrieval':generate_dataset_kv_retrieval,
        'passage_retrieval':generate_dataset_passage_retrieval,
        'passage_count':generate_dataset_passage_count,
        'counting_stars':generate_dataset_counting_stars,
}
rows = 200
out_path = '/users/PDS0352/wyang107/project/LCEG/longbench_pro/data2'
for length in tqdm([64, 126]): #
    for key in func.keys():
        result = func[key](length, rows)
        with open(os.path.join(out_path, f'{key}_{length}.jsonl'), "w", encoding="utf-8") as f:
            for pred in result:
                json.dump(pred, f, ensure_ascii=False)
                f.write('\n')



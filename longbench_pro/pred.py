import os
from datasets import load_dataset
import torch
import json
import transformers
from tqdm import tqdm
import numpy as np
import random
import argparse
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.dirname(current_dir)
sys.path.append(models_dir)

def set_global_path(path):
    return os.path.join('/users/PDS0352/wyang107/project/LCEG/longbench_pro', path)

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='llama-3.2-3B-Instruct')
    parser.add_argument('--dataset_name', type=str, default="samsum")
    return parser.parse_args(args)

def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name, preds=[] ,out_path=''):
    data = data['test']
    k = 0
    for json_obj in tqdm(data):
        context_length = json_obj["length"]
        try:
            for i, inp in enumerate(json_obj['input']):
                flag=False
                if k < len(preds):
                    pred, flag = preds[k], True
                    k = k+1
                    if len(pred['answers']) != 0:
                        continue
                else: k = k+1
                obj = {'context':json_obj['new_context'], 'input':inp, "length":json_obj["length"],
                        'answers':json_obj['answers'][i], 'instruction':json_obj['instruction']}
                prompt = prompt_format.format(**obj)
                # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
                tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
                if "chatglm3" in model_name:
                    tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
                if len(tokenized_prompt) > max_length:
                    half = int(max_length/2)
                    prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)

                input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
                context_length = input.input_ids.shape[-1]
                kwargs = {}
                kwargs['use_cache'] = True
                if model_name == "llama2-7b-hf-slimpajama-landmark" or model_name == "llama2-7b-hf-slimpajama-landmark-test4k":  
                    kwargs['offload_cache_to_cpu'] = False
                    kwargs['use_flash'] = False
                    kwargs['cache_top_k'] = 5
                if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
                    output = model.generate(
                        **input,
                        max_new_tokens=max_gen,
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0,
                        min_length=context_length+1,
                        eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                        **kwargs,
                    )[0]
                else:
                    output = model.generate(
                        **input,
                        max_new_tokens=max_gen,
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0,
                        **kwargs,
                    )[0]
                pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
                # preds.append({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": context_length})
                if flag: preds[k-1] = {"pred": pred, "answers": obj["answers"], "length": context_length}
                else: preds.append({"pred": pred, "answers": obj["answers"], "length": context_length})
        except Exception as e:
            print(f"Exception occurred: {e}")
            if flag: preds[k-1] = {"answers":'', "length": json_obj["length"]}
            else: preds.append({"answers":'', "length": json_obj["length"]})
        save_preds(out_path, preds)
    return preds

def save_preds(out_path, preds):
    with open(out_path, "w", encoding="utf-8") as f:
        for pred in preds:
            json.dump(pred, f, ensure_ascii=False)
            f.write('\n')

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name, device, use_flash_attention_2=False):
    print('testing:', model_name)
    print('model path:', path)
    cache_dir = '/users/PDS0352/wyang107/project/LCEG/model_cache'
    token='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'
    if model_name == "llama2-7b-hf" or model_name == "llama2-7b-hf-slimpajama-pi-32k" or model_name == "llama2-7b-hf-slimpajama-longlora-32k":
        config = transformers.AutoConfig.from_pretrained(path)
        print('rope_scaling:', config.rope_scaling)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            path,
            config=config,
            torch_dtype=torch.float16,
            use_flash_attention_2=use_flash_attention_2,
            device_map="auto",
            cache_dir=cache_dir,
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path,
            config=config,
            cache_dir=cache_dir,
        )
    elif model_name == "llama2-7b-hf-slimpajama-ntk-32k":
        config = transformers.AutoConfig.from_pretrained(path)
        print('rope_scaling:', config.rope_scaling)
        from models.llama_ntk_32k import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(
            path,
            config=config,
            torch_dtype=torch.float16,
            use_flash_attention_2=use_flash_attention_2,
            device_map="auto",
            cache_dir=cache_dir,
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path,
            config=config,
            cache_dir=cache_dir,
        )
    elif model_name == "llama2-7b-hf-slimpajama-ntk-64k" or model_name == "llama-2-7b-hf-slimpajama-ntk-64k-2B":
        config = transformers.AutoConfig.from_pretrained(path)
        print('rope_scaling:', config.rope_scaling)
        from models.llama_ntk_64k import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(
            path,
            config=config,
            torch_dtype=torch.float16,
            use_flash_attention_2=use_flash_attention_2,
            device_map="auto",
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path,
            config=config,
        )
    elif model_name == "llama2-7b-hf-lminfinite":
        from models.llama_infinite import LlamaForCausalLM
        from models.llama_infinite.llama import convert_llama_model
        model = LlamaForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=cache_dir,
        )
        model = convert_llama_model(model, 4096, 10)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path,
            cache_dir=cache_dir,
        )
    elif model_name == "llama2-7b-hf-ntk-frozen":
            # Set RoPE scaling factor
        config = transformers.AutoConfig.from_pretrained(
            path,
        )
        
        scaling_factor = 2.0
        print(config.rope_scaling)
        config.rope_scaling = {"type": "dynamic", "factor": scaling_factor}

        # Load model and tokenizer
        model = transformers.AutoModelForCausalLM.from_pretrained(
            path,
            config=config,
            torch_dtype=torch.float16,
            use_flash_attention_2=use_flash_attention_2,
            device_map="auto",
            cache_dir=cache_dir,
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path,cache_dir=cache_dir,
        )
        
    elif model_name == "llama2-7b-hf-slimpajama-yarn-32k":
        from models.llama_yarn.modeling_llama_yarn import LlamaForCausalLM
        from models.llama_yarn.configuration_llama import LlamaConfig
        config_cls = LlamaConfig
        model_cls = LlamaForCausalLM
        config = config_cls.from_pretrained(path)
        print(config.rope_scaling)
        model = model_cls.from_pretrained(
            path,
            config=config,
            torch_dtype=torch.float16,
            use_flash_attention_2=use_flash_attention_2,
            device_map="auto",
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path,cache_dir=cache_dir,
        )
    elif model_name == "llama2-7b-hf-selfextend":
        from transformers import AutoModelForCausalLM
        from models.llama_selfextend import SelfExtend
        window_size = 1024
        group_size = 64
        use_flash = use_flash_attention_2
        model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", torch_dtype=torch.bfloat16, use_flash_attention_2=use_flash,cache_dir=cache_dir,)
        print(f'using group size {group_size} using window size {window_size}')
        SelfExtend.apply(model, group_size, window_size, enable_flash_attention=use_flash, flash_attention_impl="flash_attn") ## flash_attention_impl="triton" or "flash_attn"
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path,cache_dir=cache_dir,
        )
    elif model_name == "llama2-7b-hf-slimpajama-clex-32k":
        print('eval clex')
        from models.llama_clex import LlamaForCausalLM, CLEXLlamaConfig
        config = CLEXLlamaConfig.from_pretrained(path)
        config.log_scale = True
        config.use_flashattn = True
        print(config.rope_scaling, flush=True)
        model = LlamaForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            config=config,
            attn_implementation="flash_attention_2",
            device_map="auto",
            cache_dir=cache_dir
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path,cache_dir=cache_dir,
        )

    elif model_name == "llama2-7b-hf-slimpajama-landmark":
        from models.llama_landmark.llama_mem import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path,
            padding_side="right",
            use_fast=False,
            cache_dir=cache_dir,
        )
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token 
        
        mem_id = tokenizer.convert_tokens_to_ids("<landmark>")
        model.set_mem_id(mem_id)
        model = model.to(device)
    else:
        config = transformers.AutoConfig.from_pretrained(path,token=token)
        print('rope_scaling:', config.rope_scaling)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path,
            config=config,
            cache_dir=cache_dir,
            token=token
        )
        if "llama-3" in model_name.lower():
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = transformers.AutoModelForCausalLM.from_pretrained(
            path,
            config=config,
            torch_dtype=torch.float16,
            use_flash_attention_2=use_flash_attention_2,
            device_map="auto",
            cache_dir=cache_dir,
            token=token
        )

    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    model2path = json.load(open(set_global_path("./config/model2path.json"), "r"))
    model2maxlen = json.load(open(set_global_path("./config/model2maxlen.json"), "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    # define your model
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device, True)
    # model, tokenizer = None, None
    max_length = model2maxlen[model_name]
    print('max_length:', max_length)

    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open(set_global_path("./config/dataset2prompt.json"), "r"))
    dataset2maxlen = json.load(open(set_global_path("./config/dataset2maxlen.json"), "r"))
    # predict on each dataset
    if not os.path.exists(set_global_path("pred/")):
        os.makedirs(set_global_path("pred/"))
    if not os.path.exists(set_global_path("pred_e")):
        os.makedirs(set_global_path("pred_e"))
    dataset_path = set_global_path("data")
    dataset = args.dataset_name
    file_names = [p.split('.')[0] for p in os.listdir(dataset_path)]
    for dataset in file_names:
        print('testing:', dataset)
        # data = load_dataset(os.path.join(dataset_path, f'{dataset}.jsonl'))
        # data = load_dataset(dataset_path, dataset, split='test', cache_dir='/users/PDS0352/wyang107/project/LCEG/model_cache/data')
        # data = load_dataset('json', data_files={'test': os.path.join(dataset_path, f'{dataset}.jsonl')})
        data = []
        with open(os.path.join(dataset_path, f'{dataset}.jsonl'), "r", encoding="utf-8") as f:
            for line in f:
                l = json.loads(line)
                data.append(l)
        data = {'test':data}
        if not os.path.exists(set_global_path(f"pred/{model_name}")):
            os.makedirs(set_global_path(f"pred/{model_name}"))
        out_path = set_global_path(f"pred/{model_name}/{dataset}.jsonl")
        key = '_'.join(dataset.split('_')[:-1])
        prompt_format = dataset2prompt[key]
        max_gen = dataset2maxlen[key]
        preds=[]
        if os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as f:
                for line in f:
                    l = json.loads(line)
                    preds.append(l)
        preds = get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name, preds, out_path)
        save_preds(out_path, preds)
#!/usr/bin/env python3
import os
import subprocess
import gzip
import json
import requests

DATA_DIR = 'human_eval_data'
DATA_FILE = os.path.join(DATA_DIR, 'HumanEval.jsonl.gz')
if not os.path.exists(DATA_FILE):
    os.makedirs(DATA_DIR, exist_ok=True)
    subprocess.run([
        'curl', '-L',
        'https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz',
        '-o', DATA_FILE
    ], check=True)

def load_human_eval(path=DATA_FILE):
    with gzip.open(path, 'rb') as f:
        return {f"HumanEval/{i}": json.loads(line) for i, line in enumerate(f)}

if __name__ == '__main__':
    problems = load_human_eval()
    print(f"Loaded {len(problems)} problems")

    samples = []
    for task_id, problem in list(problems.items()):
        # prompt = f"You are a Python expert. First outline your reasoning in comments, then write the function. Complete this Python function:\n\n{problem['prompt']}"
        prompt=problem['prompt']
        payload = {
            "model": "Qwen2.5-Coder-0.5B-Instruct-GGUF",
            "messages": [
                {"role": "system", "content": "You are a Python code generator. Provide only the complete function implementation in Python (signature and body) without any explanations or commentary."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1024,
            "temperature": 0.1,
        }

        r = requests.post("http://localhost:8080/v1/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
        content = data['choices'][0]['message']['content']
        samples.append({"task_id": task_id, "completion": content})

    # Write out completions
    with open('samples.jsonl', 'w') as fw:
        for s in samples:
            fw.write(json.dumps(s) + '\n')

    print("Generated samples.jsonl")
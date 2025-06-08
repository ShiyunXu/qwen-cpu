import json
from human_eval.evaluation import evaluate_functional_correctness
import re

def load_samples(path='samples.jsonl'):
    samples = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            code = entry['completion']
            # remove markdown fences and any language tags
            code = re.sub(r"```(?:python)?", "", code)
            # strip leading/trailing whitespace
            code = code.strip()
            samples.append({
                'task_id': entry['task_id'],
                'completion': code
            })
    return samples

if __name__ == '__main__':
    samples = load_samples()  # all samples
    eval_file = 'samples_eval.jsonl'
    with open(eval_file, 'w') as fw:
        for s in samples:
            fw.write(json.dumps(s) + "\n")

    result = evaluate_functional_correctness(
        sample_file=eval_file,
        problem_file='human_eval_data/HumanEval.jsonl.gz',
        k=[1],
        n_workers=4,
        timeout=10.0
    )
    print(f"Final Pass@1: {result['pass@1']*100:.1f}%")
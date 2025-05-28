from vllm import LLM, SamplingParams
from human_eval.data import read_problems
from tqdm import tqdm
import json

llm = LLM(
    model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
    trust_remote_code=True,
    enforce_eager=True,
)

sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.95,
    max_tokens=1024,
    stop=["<|endoftext|>", "\nclass", "\ndef"]
)

problems = read_problems()
samples = []

for task_id, problem in tqdm(problems.items()):
    prompt = f"Complete this Python function:\n\n{problem['prompt']}"
    output = llm.generate(prompt, sampling_params)
    completion = problem["prompt"] + output[0].outputs[0].text
    samples.append({"task_id": task_id, "completion": completion})

with open("samples.jsonl", "w") as f:
    for s in samples:
        f.write(json.dumps(s) + "\n")

from human_eval.evaluation import evaluate_functional_correctness

results = evaluate_functional_correctness("samples.jsonl")
print(f"Final Pass@1 Score: {results['pass@1'] * 100:.2f}%")

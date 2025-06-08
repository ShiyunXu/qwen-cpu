# Example

Step 0: Install the docker

```python
pip install docker
```

Step 1: Build environment

```python
git clone https://github.com/ShiyunXu/qwen-cpu.git
cd ./qwen-cpu
docker build -t qwen-cpu .
python3 serve_docker.py
```

Step 2: Open a new terminal and install evaluation dataset and metrics from huamn-eval. Then run inference.py and evaluate.py
```python
cd ./qwen-cpu
pip install git+https://github.com/openai/human-eval.git
python3 inference.py
python3 evaluate.py 
```

# Results

Final Pass@1: 59.1%


## Reproduction

To reproduce the result, simply run:
```python
python3 evaluate.py 
```

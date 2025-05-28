from vllm import LLM, AsyncLLMEngine, EngineArgs

engine_args = EngineArgs(
    model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
    tokenizer=None,
    dtype="float16",
    trust_remote_code=True,
    max_model_len=2048,
    enforce_eager=True,
)

engine = AsyncLLMEngine.from_engine_args(engine_args)

import asyncio
asyncio.get_event_loop().run_forever()

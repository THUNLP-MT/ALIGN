import sys
import traceback
import time
import openai
import os
import random

import datetime
VLLM_CONFIG = eval(os.getenv("VLLM_CONFIG", "[('127.0.0.1', 8000), ('127.0.0.1', 8001)]"))

clients = [
    openai.OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "sk-xxx"),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        max_retries=1,
        timeout=300
    )
]

vllm_clients = [
    openai.OpenAI(
        api_key="token-abc123",
        base_url=f"http://{url}:{port}/v1",
        max_retries=1,
        timeout=200
    ) for url, port in VLLM_CONFIG
]

model_path = {
    "Qwen2.5-7B-Instruct": "your_model_path",
}

def call_llm(
    messages,
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=8192,
    max_retries=20,
    stop=None,
    llm_port_idx=None
):
    if model in model_path:
        if llm_port_idx is not None:
            if model == "Qwen2.5-7B-Instruct":
                client = vllm_clients[llm_port_idx]
        else:
            if model == "Qwen2.5-7B-Instruct":
                client = random.choice(vllm_clients)
    else:
        client_idx = 0
        client = clients[client_idx]
    while True:
        try:
            if stop is not None:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop
                )
                content = response.choices[0].message.content
            else:
                if model in model_path:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    content = response.choices[0].message.content
                else:
                    print(f"{datetime.datetime.now()} Calling LLM with model {model}...")
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    if response.usage.prompt_tokens > 30000:
                        exit(1)
                    content = response.choices[0].message.content
                    print(f"{datetime.datetime.now()} LLM call completed.")
            return content
        except Exception as e:
            if "This model's maximum context length is " in str(e):
                raise e
            print(e)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            error_details = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            print(error_details)
            time.sleep(10)
            if model not in model_path:
                client_idx = (client_idx + 1) % len(clients)
                print(f"Retrying with client {client_idx}...")
                client = clients[client_idx]
            else:
                if model == "Qwen2.5-7B-Instruct":
                    client = random.choice(vllm_clients)
            max_retries -= 1
            if max_retries <= 0:
                raise e

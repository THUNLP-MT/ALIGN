import importlib
import json
import random
import os
import sys
import yaml
import logging
import io
from tqdm import tqdm
import concurrent.futures

import multiprocessing
from env_webshop import webshopEnv

AGENTIC_SYSTEM_DEFAULT_MODEL = os.getenv("AGENTIC_SYSTEM_DEFAULT_MODEL", "Qwen2.5-7B-Instruct")
MAX_WORKERS = 128  # Default number of parallel threads

if AGENTIC_SYSTEM_DEFAULT_MODEL=="Qwen2.5-7B-Instruct":
    from call_llm import VLLM_CONFIG
else:
    VLLM_CONFIG = [1]

from call_llm import call_llm

def run_single_task(split, file_lock, task_info, InferRules, WrapStep, logger_base_dir=None, task_logger_file_path=None, llm_port_idx=None, base_dir="webshop"):
    task_idx, split = task_info

    url = f"fixed_{task_idx}"

    if logger_base_dir:
        if os.path.exists(f"{logger_base_dir}/task_{split}_{task_idx}.json"):
            with open(f"{logger_base_dir}/task_{split}_{task_idx}.json", "r") as f:
                result = json.load(f)
            if result["success"]:
                print(f"Task {task_idx} already completed. Skipping.")
                return result
            
    print(f"{task_idx} - {split} - {url}")

    if task_logger_file_path:
        task_logger = logging.getLogger(f"task_{task_info[0]}_{task_info[1]}")
        task_logger.setLevel(logging.INFO)
        task_logger.propagate = False
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        file_handler = logging.FileHandler(task_logger_file_path)
        for handler in task_logger.handlers[:]:
            task_logger.removeHandler(handler)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        task_logger.addHandler(file_handler)
    else:
        task_logger = None

    if task_logger:
        task_logger.info(f"========== Task ID: {task_idx} ==========")

    env = webshopEnv(url)
    obs, reward, done = env.step("reset")
    init_obs = obs
    task = init_obs.split("Instruction:")[1].split("[Search]")[0].strip()

    messages = [
        {
            "role": "system",
            "content": f"""You are an AI assistant solving tasks in a webshop environment. Your goal is to help human buy the item they need.

# Action Space

In this environment, you have a set of high-level actions at your disposal, each corresponding to a typical household activity. These actions are:

Type	Argument	    State → Next State
search	[Query]	        Search → Results
click	Back to Search	* → Search
click	Prev/Next Page	Results → Results
click	[Product Title]	Results → Item
click	[Option]	    Item → Item
click	Desc/Overview	Item-Detail → Item
click	Buy	            Item → Episode End

Each action use method is `Type[Argument]`, like `search[shirt]` etc.

Although each action may internally consist of multiple embodied steps (e.g., walking to the sink, turning a knob, etc.), from your perspective you need only provide one high-level action at a time.

# Instructions

Single Action per Turn
At each step, you must respond with exactly one action (i.e., the next “thought”). Use the format:
ACTION[Argument]
ACTION[Argument]
For example:
search[shirt]
or
click[Next Page]

Environment Feedback
After you provide your single action, the environment will automatically execute it and return the resulting observation. You then decide on your next action based on the updated state.

Reasoning (Chain of Thought)
You may use hidden reasoning to figure out the best next step. However, only output the single action that represents your decision. Do not reveal your entire chain of thought.

Continue Until Task Completion
You will iterate this process—receiving the environment’s feedback, deciding on the next action, and outputting a single action—until the task is finished.

# Environment Rule

{InferRules(init_obs, task)}"""
        },
        {
            "role": "user",
            "content": f"""# Task

{obs}

Begin by examining the environment or taking any initial steps you find relevant. Remember, provide only one action each time."""
        }
    ]

    if task_logger:
        task_logger.info(f"Task: {obs}")

    log_stream = io.StringIO()
    stream_handler = logging.StreamHandler(log_stream)
    stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    function_logger = logging.getLogger(f"function_logger_{split}_{task_idx}")
    for handler in function_logger.handlers[:]:
        function_logger.removeHandler(handler)
    function_logger.setLevel(logging.DEBUG)
    function_logger.addHandler(stream_handler)
    function_logger.propagate = False

    for i in range(30):
        agent_action = call_llm(messages, model=AGENTIC_SYSTEM_DEFAULT_MODEL, temperature=0.0, max_tokens=1024, llm_port_idx=llm_port_idx)
        messages.append({"role": "assistant", "content": agent_action})
        if task_logger:
            task_logger.info(f"Agent Action: {agent_action}")

        log_stream.seek(0)
        log_stream.truncate(0)
        obs, reward, done = WrapStep(env, init_obs, task, agent_action, function_logger)
        log_content = log_stream.getvalue()

        if task_logger:
            task_logger.info(f"Observation: {obs}")
            task_logger.info(f"Reward: {reward}")
            task_logger.info(f"Done: {done}")
            if log_content:
                task_logger.info(f"Log contents when executing `WrapStep`: {log_content}\n")
            task_logger.info(f"---------------------------------")

        messages.append({
    "role": "user",
    "content": f"""# Observation from the environment
{obs}

{task}

Now you need to give your next action."""
})

        if done:
            break
    
    final_result = {'task': url, 'score': reward, 'success': True}

    if split=="train" and file_lock is not None:
        pass

    return final_result

def run_experiment_parallel(split, interface_module_name, logger_base_dir=None, _slice=None, random_choice=False, max_workers=MAX_WORKERS, base_dir=""):
    print(f"interface_module_name: {interface_module_name}")
    print(f"Using {max_workers} parallel workers")

    if logger_base_dir:
        os.makedirs(logger_base_dir, exist_ok=True)
    
    # Dictionary to store all tasks
    all_tasks = []

    if isinstance(interface_module_name, str):
        try:
            module = importlib.import_module(interface_module_name)
            InferRules = getattr(module, "InferRules")
            WrapStep = getattr(module, "WrapStep")
        except ImportError:
            print(f"Error: Could not import module '{interface_module_name}'")
            raise
        except AttributeError:
            print(f"Error: Could not find function 'get_environment_explanation' or 'WrapStep' in module '{interface_module_name}'")
            raise
    # elif isinstance(interface_module_name, FunctionType):
    #     WrapStep = interface_module_name
    else:
        raise ValueError("interface_module_name must be a string")
    
    if split=="test":
        all_tasks = [(task_idx, split) for task_idx in range(50)]
    elif split=="train":
        all_tasks = [(task_idx, split) for task_idx in range(50, 200)]
    else:
        raise ValueError("split must be 'train' or 'test'")
    
    if _slice:
        if random_choice:
            all_tasks = random.sample(all_tasks, _slice)
        else:
            all_tasks = all_tasks[:_slice]

    results = []
    
    manager = multiprocessing.Manager()
    file_lock = manager.Lock()

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {}
        for i, task_info in enumerate(all_tasks):
            if logger_base_dir:
                task_logger_file_path = f"{logger_base_dir}/task_{split}_{task_info[0]}.log"
            else:
                task_logger_file_path = None
            future_to_task[executor.submit(run_single_task, split, file_lock, task_info, InferRules, WrapStep, logger_base_dir, task_logger_file_path, i%len(VLLM_CONFIG), base_dir)] = task_info
        
        completed = 0
        total = len(future_to_task)
        pbar = tqdm(total=total, desc="Processing tasks")

        for future in concurrent.futures.as_completed(future_to_task):
            task_info = future_to_task[future]
            task_type_idx, split = task_info

            result = future.result()
            if logger_base_dir:
                result_path = f"{logger_base_dir}/task_{split}_{task_type_idx}.json"
                try:
                    with open(result_path, 'w') as f:
                        json.dump(result, f, indent=2)
                    print(f"Saved result to {result_path}")
                except Exception as e:
                    print(f"ERROR: Could not save result to {result_path}: {str(e)}")
            
            results.append(result)

            completed += 1
            pbar.update(1)
            
            # 定期保存整体结果
            if completed % 20 == 0:
                save_and_print_results(results, split, logger_base_dir)
        
        pbar.close()

    print("All tasks completed. Saving final results...")
    save_and_print_results(results, split, logger_base_dir)

    return results

def save_and_print_results(results, split, logger_base_dir):
    if not logger_base_dir:
        return

    score = {}
    score["average"] = []
    score["score"] = []
    total_score = 0
    total_count = 0

    for result in results:
        if result["success"]:
            score["score"].append(result["score"])
            total_score += result["score"]
            total_count += 1
    
    if total_count > 0:
        score["average"] = [total_score / total_count]
    
    # Save scores to file
    with open(f"{logger_base_dir}/score.json", "w") as f:
        json.dump(score, f, indent=2)
    
    # Print current results
    print(json.dumps(score, indent=2))
    print(f"Total tasks completed: {total_count}; Total score: {total_score}")
    print(f"Current average score: {total_score / total_count if total_count > 0 else 'N/A'}")
    
    # Save full results
    with open(f"{logger_base_dir}/all_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    # run_experiment_parallel("test", "env_rule", logger_base_dir="webshop/logs/baseline_vanilla", _slice=None, random_choice=False, max_workers=MAX_WORKERS, base_dir="")

    import argparse
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument("--interface_module", type=str, help="Module name for interface")
    parser.add_argument("--logger_base_dir", type=str, help="Base directory for logging results")

    argparse_args = parser.parse_args()
    interface_module = argparse_args.interface_module
    logger_base_dir = argparse_args.logger_base_dir

    run_experiment_parallel(
        split="test",
        interface_module_name=interface_module,
        logger_base_dir=logger_base_dir,
    )
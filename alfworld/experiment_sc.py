import importlib
import json
import random
from types import FunctionType
import os
import sys
import yaml
import logging
import io
from tqdm import tqdm
import concurrent.futures

import multiprocessing

os.environ["ALFWORLD_DATA"] = "alfworld/data"
AGENTIC_SYSTEM_DEFAULT_MODEL = os.getenv("AGENTIC_SYSTEM_DEFAULT_MODEL", "Qwen2.5-7B-Instruct")
MAX_WORKERS = 128  # Default number of parallel threads

if AGENTIC_SYSTEM_DEFAULT_MODEL=="Qwen2.5-7B-Instruct":
    from call_llm import VLLM_CONFIG
else:
    VLLM_CONFIG = [1]

from call_llm import call_llm
from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
class SingleAlfredTWEnv(AlfredTWEnv):
    def __init__(self, config, name, train_eval="train"):
        self.config = config
        self.train_eval = train_eval

        self.goal_desc_human_anns_prob = self.config['env']['goal_desc_human_anns_prob']
        self.get_game_logic()

        self.random_seed = 42

        self.game_files = [name]
        self.num_games = 1

def run_single_task(split, file_lock, task_info, InferRules, WrapStep, logger_base_dir=None, task_logger_file_path=None, llm_port_idx=None, base_dir="alfworld"):
    task_type_idx, task_idx, file_name, split, alfworld_config = task_info

    if logger_base_dir:
        if os.path.exists(f"{logger_base_dir}/task_{split}_{task_type_idx}_{task_idx}.json"):
            with open(f"{logger_base_dir}/task_{split}_{task_type_idx}_{task_idx}.json", "r") as f:
                result = json.load(f)
            if result["success"]:
                print(f"Task {task_type_idx}-{task_idx} already completed. Skipping.")
                return result

    print(f"{task_type_idx} - {task_idx} - {file_name} - {split}")

    if task_logger_file_path:
        if os.path.exists(task_logger_file_path):
            with open(task_logger_file_path, "w") as f:
                f.write("")
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
        task_logger.info(f"========== Task ID: {task_type_idx}-{task_idx} ==========")
    
    # with env_init_lock:
    env = SingleAlfredTWEnv(alfworld_config, file_name, split)
    env = env.init_env(batch_size=1)
    obs, info = env.reset()
    obs = '\n'.join(obs[0].split('\n\n')[1:])
    task = obs.split('\n')[1].strip()
    init_obs = obs.split('\n')[0].strip()

    messages = [
        {
            "role": "system",
            "content": f"""You are an AI assistant solving tasks in a household environment. Your goal is to break down complex tasks into simple steps and plan your actions accordingly.

# Action Space

In this environment, you have a set of high-level actions at your disposal, each corresponding to a typical household activity. These actions are:

- look:                             look around your current location
- inventory:                        check your current inventory
- go to (receptacle):               move to a receptacle
- open (receptacle):                open a receptacle
- close (receptacle):               close a receptacle
- take (object) from (receptacle):  take an object from a receptacle
- move (object) to (receptacle):    place an object in or on a receptacle
- examine (something):              examine a receptacle or an object
- use (object):                     use an object
- heat (object) with (receptacle):  heat an object using a receptacle
- clean (object) with (receptacle): clean an object using a receptacle
- cool (object) with (receptacle):  cool an object using a receptacle
- slice (object) with (object):     slice an object using a sharp object

Although each action may internally consist of multiple embodied steps (e.g., walking to the sink, turning a knob, etc.), from your perspective you need only provide one high-level action at a time.

# Instructions

Single Action per Turn
At each step, you must respond with exactly one action (i.e., the next “thought”). Use the format:
ACTION [object/receptacle specifier]
ACTION [object/receptacle specifier]
For example:
take apple from table
or
go to kitchen.

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
    function_logger = logging.getLogger(f"function_logger_{split}_{task_type_idx}_{task_idx}")
    for handler in function_logger.handlers[:]:
        function_logger.removeHandler(handler)
    function_logger.setLevel(logging.DEBUG)
    function_logger.addHandler(stream_handler)
    function_logger.propagate = False

    same_action_count = 0
    same_action = ""

    for i in range(100):
        agent_action2count = {}
        for _ in range(3):
            agent_action = call_llm(messages, model=AGENTIC_SYSTEM_DEFAULT_MODEL, temperature=0.3, max_tokens=1024, llm_port_idx=llm_port_idx)
            agent_action2count[agent_action] = agent_action2count.get(agent_action, 0) + 1
        agent_action = max(agent_action2count, key=agent_action2count.get)
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
        
        if agent_action == same_action:
            same_action_count += 1
        else:
            same_action_count = 0
            same_action = agent_action
        if same_action_count > 6:
            done = True

        if done:
            break

    final_result = {'task': file_name, 'score': int(reward), 'success': True}
    
    if split=="train" and file_lock is not None:
        gold_action_obs_sequence = []
        with file_lock:
            with open(f"{base_dir}/golden_action_obs.json", "r", encoding="utf-8") as f:
                gold_action_obs = json.load(f)
            if f"{task_type_idx}-{task_idx}" in gold_action_obs:
                # results[-1]["gold_action_obs_sequence"] = gold_action_obs[f"{task_type_idx}-{task_idx}"]
                pass
            else:
                env = SingleAlfredTWEnv(alfworld_config, file_name, split)
                env = env.init_env(batch_size=1)
                obs, info = env.reset()
                obs = '\n'.join(obs[0].split('\n\n')[1:])
                gold_action_obs_sequence.append(f"Task: {obs}")
                max_steps = 150
                done = False
                step = 0
                while not done:
                    action = info["extra.expert_plan"][0][0]
                    obs, score, done, info = env.step([action])
                    obs, reward, done = obs[0], info['won'][0], done[0]
                    gold_action_obs_sequence.append(f"Agent Action: {action}")
                    gold_action_obs_sequence.append(f"Observation: {obs} | Reward: {reward} | Done: {done}")
                    step += 1
                    if step >= max_steps or done:
                        break
                if done and info["won"][0]:
                    # results[-1]["gold_action_obs_sequence"] = gold_action_obs_sequence
                    gold_action_obs[f"{task_type_idx}-{task_idx}"] = gold_action_obs_sequence
                else:
                    # results[-1]["gold_action_obs_sequence"] = ["We can't give you the gold action sequence for this task."]
                    gold_action_obs[f"{task_type_idx}-{task_idx}"] = ["We can't give you the gold action sequence for this task."]
            with open(f"{base_dir}/golden_action_obs.json", "w", encoding="utf-8") as f:
                json.dump(gold_action_obs, f, ensure_ascii=False, indent=4)

    return final_result

def run_experiment_parallel(split, interface_module_name, logger_base_dir=None, _slice=None, random_choice=False, max_workers=MAX_WORKERS, task_type_list=[0,1,2,3,4,5], base_dir=""):
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
    
    with open(f"alfworld/file_names_{split}.json", "r") as f:
        file_names = json.load(f)
    with open("alfworld/alfworld/configs/base_config.yaml", "r") as f:
        alfworld_config = yaml.safe_load(f)

    for i in task_type_list:
        file_names[i] = [(idx, file_name) for idx, file_name in enumerate(file_names[i])]
    if _slice is not None:
        for i in task_type_list:
            if not random_choice:
                file_names[i] = file_names[i][:_slice]
            else:
                file_names[i] = random.sample(file_names[i], _slice)
    
    for i in task_type_list:
        for task_idx, file_name in file_names[i]:
            all_tasks.append((i, task_idx, file_name, split, alfworld_config))
    
    results_by_type = {i: {split: []} for i in range(6)}

    manager = multiprocessing.Manager()
    file_lock = manager.Lock()

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {}
        for i, task_info in enumerate(all_tasks):
            if logger_base_dir:
                task_logger_file_path = f"{logger_base_dir}/task_{split}_{task_info[0]}_{task_info[1]}.log"
            else:
                task_logger_file_path = None
            future_to_task[executor.submit(run_single_task, split, file_lock, task_info, InferRules, WrapStep, logger_base_dir, task_logger_file_path, i%len(VLLM_CONFIG), base_dir)] = task_info
        
        # 使用tqdm显示进度
        completed = 0
        total = len(future_to_task)
        pbar = tqdm(total=total, desc="Processing tasks")
        
        for future in concurrent.futures.as_completed(future_to_task):
            task_info = future_to_task[future]
            task_type_idx, task_idx, file_name, split, _ = task_info

            result = future.result()
            if logger_base_dir:
                result_path = f"{logger_base_dir}/task_{split}_{task_type_idx}_{task_idx}.json"
                try:
                    with open(result_path, 'w') as f:
                        json.dump(result, f, indent=2)
                    print(f"Saved result to {result_path}")
                except Exception as e:
                    print(f"ERROR: Could not save result to {result_path}: {str(e)}")
            
            results_by_type[task_type_idx][split].append(result)

            completed += 1
            pbar.update(1)
            
            # 定期保存整体结果
            # if completed % 20 == 0:
            #     save_and_print_results(results_by_type, split, logger_base_dir)
        
        pbar.close()

    print("All tasks completed. Saving final results...")
    save_and_print_results(results_by_type, split, logger_base_dir)

    return results_by_type

def save_and_print_results(results_by_type, split, logger_base_dir):
    if not logger_base_dir:
        return

    score = {}
    score["average"] = []
    total_score = 0
    total_count = 0

    for task_type_idx in range(6):
        if task_type_idx not in score:
            score[task_type_idx] = {}
        
        if split not in score[task_type_idx]:
            score[task_type_idx][split] = []
        
        # Calculate average score for this task type and split
        task_results = results_by_type[task_type_idx][split]
        if task_results:
            task_scores = [result["score"] for result in task_results]
            avg_score = sum(task_scores) / len(task_scores)
            score[task_type_idx][split] = [avg_score]
            
            total_score += sum(task_scores)
            total_count += len(task_scores)
    
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
        json.dump(results_by_type, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument("--interface_module", type=str, help="Module name for interface")
    parser.add_argument("--logger_base_dir", type=str, help="Base directory for logging results")
    
    argparse_args = parser.parse_args()
    interface_module = argparse_args.interface_module
    logger_base_dir = argparse_args.logger_base_dir

    run_experiment_parallel(
        split="eval_out_of_distribution",
        interface_module_name=interface_module,
        logger_base_dir=logger_base_dir,
    )

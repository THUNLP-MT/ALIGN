import importlib
import json
import random
import os
import sys
import time
import traceback
import yaml
import logging
import io
from tqdm import tqdm
import concurrent.futures

import multiprocessing

from scienceworld import ScienceWorldEnv

AGENTIC_SYSTEM_DEFAULT_MODEL = os.getenv("AGENTIC_SYSTEM_DEFAULT_MODEL", "Qwen2.5-7B-Instruct")
MAX_WORKERS = 80  # Default number of parallel threads
TASK_TIMEOUT = 60 * 60  # 60 minutes timeout in seconds
MAX_RETRIES = 1  # Maximum number of retries for a task

if AGENTIC_SYSTEM_DEFAULT_MODEL=="Qwen2.5-7B-Instruct":
    from call_llm import VLLM_CONFIG
else:
    VLLM_CONFIG = [1]

from call_llm import call_llm

def run_single_task(split, file_lock, task_info, InferRules, WrapStep, logger_base_dir=None, task_logger_file_path=None, llm_port_idx=None, base_dir="scienceworld"):
    task_num, variation, split = task_info

    if logger_base_dir:
        if os.path.exists(f"{logger_base_dir}/task_{split}_{task_num}_{variation}.json"):
            with open(f"{logger_base_dir}/task_{split}_{task_num}_{variation}.json", "r") as f:
                result = json.load(f)
            if result["success"]:
                print(f"Task {task_num}-{variation} already completed. Skipping.")
                return result

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

    try:
        print(f"{split} - {task_num} - {variation}")
        
        # with env_init_lock:
        env = ScienceWorldEnv("", "", envStepLimit=100)
        taskNames = env.get_task_names()
        taskName = taskNames[task_num]
        env.load(taskName, variation, "easy", generateGoldPath=True)

        if task_logger:
            task_logger.info(f"========== Task Name: {taskName} | Task ID: {task_num}-{variation} ==========")

        task = env.taskdescription()[18:]

        obs, info = env.reset()
        init_obs = obs
        

        done = False
        score = 0.0
        last_score = 0.0
        step = 0

        messages = [
            {
                "role": "system",
                "content": f"""You are an AI assistant solving tasks in a scientific laboratory environment (ScienceWorld). Your goal is to break down complex scientific tasks into simple steps and plan your actions to complete experiments or verify hypotheses.

# Action Space

In this environment, you have a set of high-level actions at your disposal, each corresponding to a typical household activity. These actions are:

- open OBJ                    open a container
- close OBJ                   close a container
- de/activate OBJ             activate/deactivate a device
- connect OBJ to OBJ          connect electrical components
- disconnect OBJ              disconnect electrical components
- use OBJ [on OBJ]            use a device/item
- look around                 describe the current room
- look at OBJ                 describe an object in detail
- look in OBJ                 describe a container's contents
- read OBJ                    read a note or book
- move OBJ to OBJ             move an object to a container
- pick up OBJ                 move an object to the inventory
- put down OBJ                drop an inventory item
- pour OBJ into OBJ           pour a liquid into a container
- dunk OBJ into OBJ           dunk a container into a liquid
- mix OBJ                     chemically mix a container
- go to LOC                   move to a new location
- teleport to LOC             teleport to a specific room
- eat OBJ                     eat a food
- flush OBJ                   flush a toilet
- focus on OBJ                signal intent on a task object
- wait [DURATION]             take no action for some duration
- task                        describe current task
- inventory                   list agent's inventory

Each action corresponds to a single command in the environment. Although some actions may involve multiple internal steps (e.g., walking to a lab bench, picking up a tool), you need only provide one high-level action at a time.

# Instructions

Single Action per Turn
At each step, you must respond with exactly one action (i.e., the next “thought”). Use the format:
ACTION [object/receptacle specifier]
ACTION [object/receptacle specifier]
For example:
close art studio door
or
mix hallway

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

{init_obs}

{task}

Begin by examining the environment or taking any initial steps you find relevant. Remember, provide only one action each time.

Now you need to give your thought and next action, you should respond in the format as follows:
THOUGHT: your thought in one line
ACTION: your next action in one line"""
            }
        ]

        if task_logger:
            task_logger.info(f"Task: {obs}")

        log_stream = io.StringIO()
        stream_handler = logging.StreamHandler(log_stream)
        stream_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        function_logger = logging.getLogger(f"function_logger_{split}_{task_num}_{variation}")
        for handler in function_logger.handlers[:]:
            function_logger.removeHandler(handler)
        function_logger.setLevel(logging.DEBUG)
        function_logger.addHandler(stream_handler)
        function_logger.propagate = False

        same_action = ""
        same_action_count = 0

        for i in range(100):
            agent_action = call_llm(messages, model=AGENTIC_SYSTEM_DEFAULT_MODEL, temperature=0.0, max_tokens=200, llm_port_idx=llm_port_idx)
            messages.append({"role": "assistant", "content": agent_action})
            if task_logger:
                task_logger.info(f"Agent Output: {agent_action}")
            try:
                agent_action = agent_action.split("ACTION:")[1].strip()
            except Exception as e:
                pass
            if task_logger:
                task_logger.info(f"Agent Action: {agent_action}")

            log_stream.seek(0)
            log_stream.truncate(0)
            obs, done, score = WrapStep(env, init_obs, task, agent_action, function_logger)
            log_content = log_stream.getvalue()

            if task_logger:
                task_logger.info(f"Observation: {obs}")
                task_logger.info(f"Score: {score}")
                task_logger.info(f"Done: {done}")
                if log_content:
                    task_logger.info(f"Log contents when executing `WrapStep`: {log_content}\n")
                task_logger.info(f"---------------------------------")

            messages.append({
        "role": "user",
        "content": f"""# Observation from the environment
{obs}

{task}

Now you need to give your thought and next action, you should respond in the format as follows:
THOUGHT: your thought in one line
ACTION: your next action in one line"""
    })
            
            if score < 0:
                done = True
                score = last_score
            last_score = score

            if agent_action == same_action:
                same_action_count += 1
            else:
                same_action_count = 0
                same_action = agent_action
            if same_action_count > 6:
                done = True

            if done:
                break

        final_result = {'task': f"{task_num}-{variation}", 'score': score, 'success': True}
        
        if split=="train" and file_lock is not None:
            gold_action_obs_sequence = []
            with file_lock:
                with open(f"{base_dir}/golden_action_obs.json", "r", encoding="utf-8") as f:
                    gold_action_obs = json.load(f)
                if f"{task_num}-{variation}" in gold_action_obs:
                    pass
                else:
                    env = ScienceWorldEnv("", "", envStepLimit=100)
                    taskNames = env.get_task_names()
                    taskName = taskNames[task_num]
                    env.load(taskName, variation, "easy", generateGoldPath=True)
                    gold_action_sequence = env.get_gold_action_sequence()
                    task = env.taskdescription()[18:]
                    obs, info = env.reset()
                    init_obs = obs

                    gold_action_obs_sequence.append(f"Task: {task}\n{obs}")
                    
                    for action in gold_action_sequence:
                        obs, _, done, info = env.step(action)
                        gold_action_obs_sequence.append(f"Agent Action: {action}")
                        gold_action_obs_sequence.append(f"Observation: {obs} | Score: {info['score']} | Done: {done}")
                        if done:
                            break
                    
                    gold_action_obs[f"{task_num}-{variation}"] = gold_action_obs_sequence
                with open(f"{base_dir}/golden_action_obs.json", "w", encoding="utf-8") as f:
                    json.dump(gold_action_obs, f, ensure_ascii=False, indent=4)
        return final_result
    
    except Exception as e:
        print(f"ERROR in task {task_num}-{variation}: {e}")
        traceback.print_exc() # 打印完整的错误堆栈信息
        # if task_logger:
        #     task_logger.error(f"Exception in task {task_num}-{variation}: {e}\n{traceback.format_exc()}")
        # 返回一个失败结果，而不是让异常传播导致工作进程崩溃
        return {'task': f"{task_num}-{variation}", 'score': 0.0, 'success': False, 'error': str(e)}
    finally:
        # 确保执行清理操作
        print(f"Task {task_num}-{variation} entering finally block.") # 调试信息
        if env is not None:
            # 如果 ScienceWorldEnv 未来有 close/cleanup 方法，在这里调用
            # 目前主要依赖进程退出时的自动清理
            pass
        # 确保关闭日志处理器，释放文件句柄
        if task_logger:
             print(f"Closing logger handlers for task {task_num}-{variation}") # 调试信息
             for handler in task_logger.handlers[:]:
                 try:
                     handler.close()
                 except Exception as log_close_err:
                     print(f"Error closing log handler for {task_num}-{variation}: {log_close_err}")
                 task_logger.removeHandler(handler)
        print(f"Task {task_num}-{variation} finished finally block.") # 调试信息


def run_experiment_parallel(split, interface_module_name, logger_base_dir=None, _slice=None, random_choice=False, max_workers=MAX_WORKERS, task_type_list=[i for i in range(30)], base_dir=""):
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
    else:
        raise ValueError("interface_module_name must be a string")
    
    env = ScienceWorldEnv("", "", envStepLimit=100)
    taskNames = env.get_task_names()
    for task_num in task_type_list:
        taskName = taskNames[task_num]
        env.load(taskName, 0, "easy")
        if "split"=="train":
            variations = env.get_variations_train()
        else:
            variations = env.get_variations_test()
        if _slice is not None and _slice < len(variations):
            if random_choice:
                variations = random.sample(variations, _slice)
            else:
                variations = variations[:_slice]
        for variation in variations:
            all_tasks.append((task_num, variation, split))

    results_by_type = {i: {split: []} for i in range(30)}

    manager = multiprocessing.Manager()
    file_lock = manager.Lock()

    # Track retry attempts for each task
    retry_counts = {task: 0 for task in all_tasks}
    
    # Track completed and active tasks
    completed_tasks = set()
    active_futures = {}
    
    # Main execution loop
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Initial submission of all tasks
        for i, task_info in enumerate(all_tasks):
            if logger_base_dir:
                task_logger_file_path = f"{logger_base_dir}/task_{split}_{task_info[0]}_{task_info[1]}.log"
            else:
                task_logger_file_path = None
            
            future = executor.submit(run_single_task, split, file_lock, task_info, 
                                   InferRules, WrapStep, 
                                   logger_base_dir, task_logger_file_path, 
                                   i % len(VLLM_CONFIG), base_dir)
            active_futures[future] = task_info
        
        # Set up progress bar
        total = len(all_tasks)
        pbar = tqdm(total=total, desc="Processing tasks")
        
        # Process tasks as they complete
        while active_futures:
            print(f"Active tasks: {len(active_futures)}")
            # 打印所有 value
            if len(active_futures) < 10:
                for future, task_info in active_futures.items():
                    print(f"{task_info[0]}-{task_info[1]}", end=" ")
                print()
            try:
                # Wait for any future to complete with a short timeout
                for future in concurrent.futures.as_completed(active_futures.keys(), timeout=5):
                    task_info = active_futures[future]
                    
                    try:
                        # Get result (should be immediately available since future is completed)
                        result = future.result()
                        
                        # Task completed successfully
                        task_num, variation, split = task_info
                        
                        # Save individual task result
                        if logger_base_dir:
                            result_path = f"{logger_base_dir}/task_{split}_{task_num}_{variation}.json"
                            try:
                                with open(result_path, 'w') as f:
                                    json.dump(result, f, indent=2)
                                print(f"Saved result to {result_path}")
                            except Exception as e:
                                print(f"ERROR: Could not save result to {result_path}: {str(e)}")
                        
                        # Add to results collection
                        results_by_type[task_num][split].append(result)
                        
                        # Mark task as completed
                        completed_tasks.add(future)
                        pbar.update(1)
                        
                    except Exception as e:
                        # Task failed with an exception
                        task_num, variation, split = task_info
                        print(f"Task {task_num}-{variation} failed with error: {str(e)}")
                        
                        # Create a failure result
                        failure_result = {'task': f"{task_num}-{variation}", 'score': 0.0, 'success': False, 'error': str(e)}
                        
                        # Save failure result
                        if logger_base_dir:
                            result_path = f"{logger_base_dir}/task_{split}_{task_num}_{variation}.json"
                            try:
                                with open(result_path, 'w') as f:
                                    json.dump(failure_result, f, indent=2)
                                print(f"Saved failure result to {result_path}")
                            except Exception as write_error:
                                print(f"ERROR: Could not save failure result to {result_path}: {str(write_error)}")
                        
                        # Add to results collection
                        results_by_type[task_num][split].append(failure_result)
                        
                        # Mark task as completed
                        completed_tasks.add(future)
                        pbar.update(1)
            
            except concurrent.futures.TimeoutError:
                # Check for timed-out tasks
                futures_to_check = list(active_futures.keys())
                for future in futures_to_check:
                    if future in completed_tasks:
                        continue
                        
                    task_info = active_futures[future]
                    if future.done():
                        # Future completed during our timeout check
                        continue
                        
                    # Check if task has been running too long
                    if not hasattr(future, '_start_time'):
                        # Add start time if not present
                        future._start_time = time.time()
                    
                    running_time = time.time() - future._start_time
                    if running_time > TASK_TIMEOUT:
                        # Task timed out
                        task_num, variation, split = task_info
                        retry_counts[task_info] += 1
                        
                        if retry_counts[task_info] <= MAX_RETRIES:
                            # Log the retry
                            print(f"Task {task_num}-{variation} timed out after {TASK_TIMEOUT//60} minutes. Retrying ({retry_counts[task_info]}/{MAX_RETRIES})...")
                            
                            # Cancel the existing future
                            future.cancel()
                            completed_tasks.add(future)
                            
                            # Resubmit the task
                            if logger_base_dir:
                                task_logger_file_path = f"{logger_base_dir}/task_{split}_{task_num}_{variation}.log"
                                # Append retry information to the log
                                if os.path.exists(task_logger_file_path):
                                    os.remove(task_logger_file_path)
                            else:
                                task_logger_file_path = None
                            
                            new_future = executor.submit(run_single_task, split, file_lock, task_info, 
                                                      InferRules, WrapStep, 
                                                      logger_base_dir, task_logger_file_path, 
                                                      task_info[0] % len(VLLM_CONFIG), base_dir)
                            new_future._start_time = time.time()  # Set start time for the new future
                            active_futures[new_future] = task_info
                        else:
                            # Max retries reached
                            print(f"Task {task_num}-{variation} failed after {MAX_RETRIES} retries. Marking as failed.")
                            
                            # Create a failure result
                            failure_result = {'task': f"{task_num}-{variation}", 'score': 0.0, 'success': False, 'error': 'Timed out'}
                            
                            # Save failure result
                            if logger_base_dir:
                                result_path = f"{logger_base_dir}/task_{split}_{task_num}_{variation}.json"
                                try:
                                    with open(result_path, 'w') as f:
                                        json.dump(failure_result, f, indent=2)
                                    print(f"Saved failure result to {result_path}")
                                except Exception as e:
                                    print(f"ERROR: Could not save failure result to {result_path}: {str(e)}")
                            
                            # Add to results collection
                            results_by_type[task_num][split].append(failure_result)
                            
                            # Mark task as completed
                            completed_tasks.add(future)
                            pbar.update(1)

            print(f"Completed Task Num: {len(completed_tasks)}")
            print(f"Active tasks Before Loop: {len(active_futures)}")
            # Remove completed futures from active list
            for future in completed_tasks:
                if future in active_futures:
                    del active_futures[future]
            print(f"Active tasks After Loop: {len(active_futures)}")
                    
            # Save intermediate results periodically
            # if len(completed_tasks) % 20 == 0 and completed_tasks:
            #     save_and_print_results(results_by_type, split, logger_base_dir)
            
            # Small sleep to avoid busy-waiting
            time.sleep(0.5)
        
        # 在run_experiment_parallel函数的while循环结束后添加：
        print("所有任务完成。准备关闭executor...")

        # 强制取消任何剩余的futures（如果存在）
        for future in list(active_futures.keys()):
            future.cancel()

        # 清除引用以允许垃圾回收
        active_futures.clear()
        completed_tasks.clear()

        # 显式释放多进程管理器资源
        file_lock = None
        manager.shutdown()  # 显式关闭管理器

        print("所有资源已清理。保存最终结果...")
        save_and_print_results(results_by_type, split, logger_base_dir)

        # Close progress bar after all tasks are completed
        pbar.close()

        executor.shutdown(wait=True, cancel_futures=True)

    return results_by_type

def save_and_print_results(results_by_type, split, logger_base_dir):
    if not logger_base_dir:
        return

    score = {}
    score["average"] = []
    total_score = 0
    total_count = 0

    for task_num in range(30):
        if task_num not in score:
            score[task_num] = {}
        
        if split not in score[task_num]:
            score[task_num][split] = []
        
        # Calculate average score for this task type and split
        task_results = results_by_type[task_num][split]
        if task_results:
            task_scores = [result["score"] for result in task_results]
            avg_score = sum(task_scores) / len(task_scores)
            score[task_num][split] = [avg_score]
            
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
    # run_experiment_parallel("test", interface_module_name="env_rule_action", logger_base_dir="scienceworld/logs/baseline_vanilla_2", _slice=5, random_choice=False, max_workers=MAX_WORKERS, task_type_list=[i for i in range(30)])

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
        _slice=5,
        random_choice=False,
        max_workers=MAX_WORKERS,
        task_type_list=[i for i in range(30)],
    )
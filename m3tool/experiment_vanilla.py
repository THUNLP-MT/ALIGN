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

AGENTIC_SYSTEM_DEFAULT_MODEL = os.getenv("AGENTIC_SYSTEM_DEFAULT_MODEL", "Qwen2.5-7B-Instruct")
MAX_WORKERS = 128

if AGENTIC_SYSTEM_DEFAULT_MODEL=="Qwen2.5-7B-Instruct":
    from call_llm import VLLM_CONFIG
else:
    VLLM_CONFIG = [1]

from call_llm import call_llm

from tasks import Task, task_iterators

TOOL_DESC = ['You have access to the following tools:\n[1] convert_hex_to_ascii: Converts a hexadecimal string to ASCII. Arguments: hex_string (str)\n    Signature: convert_hex_to_ascii(hex_string: str) -> str\n[2] reverse_string: Reverses a string. Arguments: string (str)\n    Signature: reverse_string(string: str) -> str\n[3] caesar_decode: Decodes a string using the Caesar cipher. Arguments: message (str), shift (int)\n    Signature: caesar_decode(message: str, shift: int) -> str\n[4] string_length: Finds the length of a string. Arguments: string (str)\n    Signature: string_length(string: str) -> int\n[5] minimum_value: Finds the minimum value from given arguments. Arguments: *args (variable number of arguments)\n    Signature: minimum_value(*args) -> int/float\n[6] maximum_value: Finds the maximum value from given arguments. Arguments: *args (variable number of arguments)\n    Signature: maximum_value(*args) -> int/float\n', 'You have access to the following tools:\n[1] count_nucleotides: Counts the occurrences of each nucleotide in a DNA sequence. Arguments: dna_sequence (str)\n    Signature: count_nucleotides(dna_sequence: str) -> dict\n[2] transcribe_dna_to_mrna: Transcribes DNA sequence to mRNA. Arguments: dna_sequence (str)\n    Signature: transcribe_dna_to_mrna(dna_sequence: str) -> str\n[3] translate_mrna_to_amino_acid: Translates mRNA sequence to a chain of amino acids. Arguments: mrna_sequence (str)\n    Signature: translate_mrna_to_amino_acid(mrna_sequence: str) -> str\n[4] find_max_nucleotide: Return the nucleotide (str) with the maximum count (int). Arguments: nucleotide_counts in the form of (k1, v1, k2, v2, ..., kn, vn)\n    Signature: find_max_nucleotide(*args) -> (str, int)\n[5] is_valid_dna_sequence: Checks if the DNA sequence is valid. Arguments: dna_sequence (str)\n    Signature: is_valid_dna_sequence(dna_sequence: str) -> bool\n[6] reverse_transcribe_mrna_to_dna: Reverse transcribes mRNA sequence to DNA. Arguments: mrna_sequence (str)\n    Signature: reverse_transcribe_mrna_to_dna(mrna_sequence: str) -> str\n', 'You have access to the following tools:\n[1] convert_currency: Converts the commodity price to local currency. Arguments: base_price (float), conversion_rate (float)\n    Signature: convert_currency(base_price: float, conversion_rate: float) -> float\n[2] calculate_tariff: Calculates the trade tariff based on the converted price. Arguments: price (float), tariff_rate (float, in %)\n    Signature: calculate_tariff(price: float, tariff_rate: float) -> float\n[3] estimate_final_value: Estimates the final trade value including the tariff. Arguments: price (float), tariff (float)\n    Signature: estimate_final_value(price: float, tariff: float) -> float\n[4] calculator: Evaluates the given expression and returns the result. Accepts a calculation expression as input. For example, "2 + (3 * 4)" will return 14.\n    Signature: calculator(expression: str) -> float\n[5] find_minimum: Finds the minimum value among the given arguments. Accepts variable number of float arguments.\n    Signature: find_minimum(*args: float) -> float\n[6] find_maximum: Finds the maximum value among the given arguments. Accepts variable number of float arguments.\n    Signature: find_maximum(*args: float) -> float\n', 'You have access to the following tools:\n[1] find_flights: Finds flights based on source, destination and date. Arguments: from_location (str), to_location (str), date (str) in YYYY-MM-DD format.\nReturns a list of flights, each represented as a dictionary with keys "from_location", "to_location" (destination), "date", and "price".\nExample: [{"from_location": "A", "to_location": "B", "date": "2023-12-25", "price": 450}]\n    Signature: find_flights(destination: str, date: str) -> List[Dict]\n[2] book_hotel: Books a hotel based on location and preferences. Arguments: location (str), *preferences (variable number of str arguments).\nReturns a list of hotels, each represented as a dictionary with keys "location", "preferences", "price_per_night", and "rating".\nExample: [{"location": "A", "preferences": ["wifi", "pool"], "price_per_night": 120, "rating": 4}]\n    Signature: book_hotel(location: str, *preferences: str) -> List[Dict]\n[3] budget_calculator: Calculates the total budget for a trip. Arguments: flight_price (float), hotel_price_per_night (float), num_nights (int).\nReturns the total budget (float).\n    Signature: budget_calculator(flight_price: float, hotel_price_per_night: float, num_nights: int) -> float\n[4] max: Finds the maximum value among the given arguments. Accepts variable number of float arguments.\n    Signature: max(*args: float) -> float\n[5] min: Finds the minimum value among the given arguments. Accepts variable number of float arguments.\n    Signature: min(*args: float) -> float\n[6] sum: Sums the given arguments. Accepts variable number of float arguments.\n    Signature: sum(*args: float) -> float\n', 'You have access to the following tools:\n[1] click_url: Clicks on a URL. A clickable URL looks like [Clickable \'<url_argument>\'] in the webpage.\nArguments: url (str).\nReturns the rendered content of the webpage after clicking the URL showing on the current rendered page.\n\n    Signature: click_url(url: str) -> str\n[2] go_to_previous_page: Goes back to the previous page. It has no arguments.\nAfter going back to the previous page, return the rendered content of the webpage.\n    Signature: go_to_previous_page() -> str\n[3] scroll_down: Scrolls down the view. It has no arguments.\nReturns the rendered content of the webpage after scrolling down.\n    Signature: scroll_down() -> str\n[4] scroll_up: Scrolls up the view. It has no arguments.\nReturns the rendered content of the webpage after scrolling up.\n    Signature: scroll_up() -> str\n[5] view: Return the current view in string format of the rendered webpage. It has no arguments.\nReturns the rendered content of the webpage.\nYou should call this when you want to see the rendered content of the current webpage.\n    Signature: view() -> str\n[6] calculator: Evaluates the given expression and returns the result. Accepts a calculation expression as input. For example, "2 + (3 * 4)" will return 14.\n    Signature: calculator(expression: str) -> float\n']

def run_single_task(split, file_lock, task_info, InferRules, WrapStep, logger_base_dir=None, task_logger_file_path=None, llm_port_idx=None, base_dir="m3tool"):
    task_type_idx, task_idx, split = task_info

    if logger_base_dir:
        if os.path.exists(f"{logger_base_dir}/task_{split}_{task_type_idx}_{task_idx}.json"):
            with open(f"{logger_base_dir}/task_{split}_{task_type_idx}_{task_idx}.json", "r") as f:
                result = json.load(f)
            if result["success"]:
                print(f"Task {task_type_idx}-{task_idx} already completed. Skipping.")
                return result

    print(f"{task_type_idx} - {task_idx} - {split}")

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

    env: Task = task_iterators[task_type_idx][0][task_idx]
    env.reset()
    task_name = env.name.strip()
    instruction = env.instruction.strip()
    
    if task_logger:
        task_logger.info(f"========== Task Name: {task_name} | Task ID: {task_type_idx}-{task_idx} ==========")

    messages = [
        {
            "role": "system",
            "content": f"""You are an AI assistant solving tasks using tools. You have access to the following tools:

{TOOL_DESC[task_type_idx]}

You can use the tools by outputing the tool name followed by its arguments, delimited by commas.
You should begin your tool invocation with 'Action:' and end it with 'End Action'.
Example: 'Action: tool_name, argument_1 End Action'
You can only invoke one tool at a time.

# Environment Rule

{InferRules(task_name, task_type_idx)}

Remember, provide only one action each time."""
        },
        {
            "role": "user",
            "content": f"""Instruction: {instruction}

If you need to output the answer, you should only respond in following format:
Answer: <your answer>

If you need to call tool, you can only invoke one tool at a time, you should only respond in following format:
Action: <your action to call tool_name> End Action
"""
        }
    ]

    if task_logger:
        task_logger.info(f"Task: {instruction}")
        task_logger.info(f"Tools Description: {TOOL_DESC[task_type_idx]}")
    
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

    for i in range(10):
        agent_action = call_llm(messages, model=AGENTIC_SYSTEM_DEFAULT_MODEL, temperature=0.0, max_tokens=1024, llm_port_idx=llm_port_idx)
        messages.append({"role": "assistant", "content": agent_action})
        if task_logger:
            task_logger.info(f"Agent Action: {agent_action}")
        
        log_stream.seek(0)
        log_stream.truncate(0)
        obs, reward, done = WrapStep(env, task_name, instruction, agent_action, function_logger)
        log_content = log_stream.getvalue()

        if task_logger:
            task_logger.info(f"Observation: {obs}")
            task_logger.info(f"Reward: {reward}")
            task_logger.info(f"Done: {done}")
            if log_content:
                task_logger.info(f"Log contents when executing `WrapStep`: {log_content}\n")
            task_logger.info(f"---------------------------------")
        
        messages.append({"role": "user", "content": f"""{obs}

If you need to output the answer, you should only respond in following format:
Answer: <your answer>

If you need to call tool, you can only invoke one tool at a time, you should only respond in following format:
Action: <your action to call tool_name> End Action"""})

        if done:
            break

    final_result = {'task': task_name, 'score': reward, 'success': True}   
    env.free_resource()

    return final_result

def run_experiment_parallel(split, interface_module_name, logger_base_dir=None, max_workers=MAX_WORKERS, task_type_list=[0,1,2,3,4], base_dir="m3tool"):
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
    
    if split=="train":
        for i in task_type_list:
            for j in range(4, 6):
                all_tasks.append((i, j, split))
    elif split=="test":
        for i in task_type_list:
            for j in range(0, 4):
                all_tasks.append((i, j, split))
            for j in range(6, task_iterators[i][1]):
                all_tasks.append((i, j, split))
    
    results_by_type = {i: {split: []} for i in range(5)}

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
        
        completed = 0
        total = len(future_to_task)
        pbar = tqdm(total=total, desc="Processing tasks")

        for future in concurrent.futures.as_completed(future_to_task):
            task_info = future_to_task[future]
            task_type_idx, task_idx, split = task_info

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

    for task_type_idx in range(5):
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
        split="test",
        interface_module_name=interface_module,
        logger_base_dir=logger_base_dir,
    )

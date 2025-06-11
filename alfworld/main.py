import glob
import os
import sys
import shutil
import traceback
import random
from tqdm import tqdm
AGENTIC_SYSTEM_DEFAULT_MODEL = os.getenv("AGENTIC_SYSTEM_DEFAULT_MODEL", "Qwen2.5-7B-Instruct")
ANALYSIS_AGENT_MODEL = os.getenv("ANALYSIS_AGENT_MODEL", "gpt-4.1-2025-04-14")
OPTIMIZATION_AGENT_MODEL_CODE = os.getenv("OPTIMIZATION_AGENT_MODEL_CODE", "gemini-2.5-pro-exp-03-25")
OPTIMIZATION_AGENT_MODEL_VALID = os.getenv("OPTIMIZATION_AGENT_MODEL_VALID", "gpt-4.1-2025-04-14")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "vanilla_gpt41_gemini25_gpt41")

TEMPLATE = os.getenv("TEMPLATE", "vanilla")

import datetime
import json
date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

base_dir = f"alfworld/logs/{EXPERIMENT_NAME}_{date_time}"

_slice = int(os.getenv("slice", 3))

train_task_list = eval(os.getenv("train_task_list", "[0, 1, 2, 3, 4, 5]"))

initial_interface_module_name = os.getenv("INTERFACE_MODULE_NAME", "interface_ini")
interface_module_name = initial_interface_module_name

environment_logics = "No Analysis Currently"

initial_turn = 0

if os.getenv("past") is not None:
    base_dir = os.getenv("past")
    with open(f"{base_dir}/config.json", "r") as f:
        config = json.load(f)
    AGENTIC_SYSTEM_DEFAULT_MODEL = config["AGENTIC_SYSTEM_DEFAULT_MODEL"]
    ANALYSIS_AGENT_MODEL = config["ANALYSIS_AGENT_MODEL"]
    OPTIMIZATION_AGENT_MODEL_CODE = config["OPTIMIZATION_AGENT_MODEL_CODE"]
    OPTIMIZATION_AGENT_MODEL_VALID = config["OPTIMIZATION_AGENT_MODEL_VALID"]
    EXPERIMENT_NAME = config["EXPERIMENT_NAME"]
    TEMPLATE = config["TEMPLATE"]
    os.environ["TEMPLATE"] = TEMPLATE
    date_time = config["date_time"]
    base_dir = config["base_dir"]
    _slice = config["slice"]
    train_task_list = config["train_task_list"]
    initial_interface_module_name = config["INTERFACE_MODULE_NAME"]

    initial_turn = 1
    if not os.path.exists(f"{base_dir}/turn_{initial_turn}/{initial_interface_module_name}_{EXPERIMENT_NAME}_{date_time}_turn_{initial_turn}.py"):
        raise Exception(f"Please run the first turn of the experiment first. The file {base_dir}/turn_{initial_turn}/{initial_interface_module_name}_{EXPERIMENT_NAME}_{date_time}_turn_{initial_turn}.py does not exist.")
    while True:
        if not os.path.exists(f"{base_dir}/turn_{initial_turn+1}/{initial_interface_module_name}_{EXPERIMENT_NAME}_{date_time}_turn_{initial_turn+1}.py"):
            break
        initial_turn += 1
    interface_module_name = f"{initial_interface_module_name}_{EXPERIMENT_NAME}_{date_time}_turn_{initial_turn}"
    with open(f"{base_dir}/turn_{initial_turn-1}/environment_logics.txt", "r") as f:
        environment_logics = f.read()
    shutil.move(
        f"{base_dir}/turn_{initial_turn}/{initial_interface_module_name}_{EXPERIMENT_NAME}_{date_time}_turn_{initial_turn}.py",
        f"alfworld/{initial_interface_module_name}_{EXPERIMENT_NAME}_{date_time}_turn_{initial_turn}.py"
    )
    # 递归删除文件夹内的所有文件
    def remove_all_files_in_directory(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isdir(file_path):
                remove_all_files_in_directory(file_path)
                os.rmdir(file_path)
            else:
                os.remove(file_path)
    if not os.path.exists(f"{base_dir}/turn_{initial_turn}/exp_logger.log"):
        remove_all_files_in_directory(f"{base_dir}/turn_{initial_turn}")

os.makedirs(base_dir, exist_ok=True)

import json
with open(f"{base_dir}/config.json", "w") as f:
    json.dump({
        "AGENTIC_SYSTEM_DEFAULT_MODEL": AGENTIC_SYSTEM_DEFAULT_MODEL,
        "ANALYSIS_AGENT_MODEL": ANALYSIS_AGENT_MODEL,
        "OPTIMIZATION_AGENT_MODEL_CODE": OPTIMIZATION_AGENT_MODEL_CODE,
        "OPTIMIZATION_AGENT_MODEL_VALID": OPTIMIZATION_AGENT_MODEL_VALID,
        "EXPERIMENT_NAME": EXPERIMENT_NAME,
        "date_time": date_time,
        "base_dir": base_dir,
        "slice": _slice,
        "train_task_list": train_task_list,
        "INTERFACE_MODULE_NAME": initial_interface_module_name,
        "TEMPLATE": TEMPLATE,
    }, f, indent=2)

if TEMPLATE=="vanilla":
    from experiment_vanilla import run_experiment_parallel
elif TEMPLATE=="react":
    from experiment_react import run_experiment_parallel
elif TEMPLATE=="sc":
    from experiment_sc import run_experiment_parallel
elif TEMPLATE=="refine":
    from experiment_refine import run_experiment_parallel
elif TEMPLATE=="planning":
    from experiment_planning import run_experiment_parallel
else:
    raise Exception(f"Unknown template: {TEMPLATE}")
from tqdm import tqdm

import logging
exp_logger = logging.getLogger(f"{EXPERIMENT_NAME}_{date_time}_experiment")
exp_logger.setLevel(logging.INFO)
exp_logger.propagate = False

agent_logger = logging.getLogger(f"{EXPERIMENT_NAME}_{date_time}_agent")
agent_logger.setLevel(logging.INFO)
agent_logger.propagate = False

from analysis_agent import AnalysisAgent
analysis_agent = AnalysisAgent()

from optimization_agent import OptimizationAgent
optimization_agent = OptimizationAgent()

if not os.path.exists(f"{base_dir}/golden_action_obs.json"):
    with open("alfworld/golden_action_obs.json", "r", encoding="utf-8") as f:
        golden_action_obs = json.load(f)
    with open(f"{base_dir}/golden_action_obs.json", "w", encoding="utf-8") as f:
        json.dump(golden_action_obs, f, indent=2)

try:
    for turn in tqdm(range(initial_turn, initial_turn + 8)):
        os.makedirs(f"{base_dir}/turn_{turn}", exist_ok=True)
        exp_logger_file = f"{base_dir}/turn_{turn}/exp_logger.log"

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        agent_logger_file = f"{base_dir}/turn_{turn}/agent_logger.log"
        for handler in agent_logger.handlers[:]:
            agent_logger.removeHandler(handler)
        file_handler = logging.FileHandler(agent_logger_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        agent_logger.addHandler(file_handler)

        score = {}
        if not os.path.exists(exp_logger_file):
            results = run_experiment_parallel(
                split="train",
                interface_module_name=interface_module_name,
                logger_base_dir=f"{base_dir}/turn_{turn}",
                _slice=_slice,
                random_choice=True,
                task_type_list=train_task_list,
                base_dir=base_dir,
            )
            # 遍历 f"{base_dir}/turn_{turn}" 目录下的所有 task_*.log 文件，将其合并到 exp_logger_file 中
            # 搜索所有 task_*.log 文件
            task_log_files = sorted(glob.glob(os.path.join(f"{base_dir}/turn_{turn}", "task_*.log")))
            
            # 以追加方式打开 exp_logger_file，将所有 log 内容合并写入
            with open(exp_logger_file, 'a', encoding='utf-8') as out_f:
                for task_file in task_log_files:
                    with open(task_file, 'r', encoding='utf-8') as in_f:
                        for line in in_f:
                            out_f.write(line)
                    out_f.write("\n")  # 添加换行符以分隔不同文件的内容
        
        with open(f"{base_dir}/golden_action_obs.json", "r", encoding="utf-8") as f:
            gold_action_obs = json.load(f)
        
        with open(f"alfworld/{interface_module_name}.py", "r") as f:
            cur_env_rule = f.read()
        
        with open(exp_logger_file, "r") as f:
            all_logging = f.readlines()
        
        env_logging = []
        nothing_happens_num = []
        slice_len = -1
        for line in all_logging:
            if line.strip() == "":
                continue
            if "INFO - ==========" in line:
                if slice_len != -1 and len(env_logging) > 0:
                    env_logging[-1]["logging"] = env_logging[-1]["logging"][:slice_len].strip()
                env_logging.append({"score": 0, "logging": ""})
                task_id = line.split("Task ID: ")[1].split(" ========")[0].strip()
                env_logging[-1]["task_id"] = task_id
                env_logging[-1]["gold_action_obs_sequence"] = gold_action_obs[task_id]
                nothing_happens_num = []
                slice_len = -1
            elif "INFO - Reward:" in line:
                env_logging[-1]["score"] = int(eval(line.split("Reward: ")[1].strip()))
            elif "INFO - Observation:" in line:
                if line.split("Observation: ")[1].strip() == "Nothing happens.":
                    nothing_happens_num.append(1)
                else:
                    nothing_happens_num.append(0)
                if len(nothing_happens_num) > 6:
                    nothing_happens_num = nothing_happens_num[-6:]
            env_logging[-1]["logging"] += line
            if len(nothing_happens_num) == 6 and sum(nothing_happens_num) > 4 and "INFO - Done:" in line and slice_len == -1:
                slice_len = len(env_logging[-1]["logging"])
        
        # 将 all_logging 随机打乱
        random.shuffle(env_logging)

        last_environment_logics = environment_logics
        cur_new_environment_logics = ""
        if not os.path.exists(f"{base_dir}/turn_{turn}/environment_logics.txt"):
            new_environment_logics = analysis_agent.analyze_logging(
                cur_env_rule=cur_env_rule,
                env_logging=env_logging,
                # model=MAIN_MODEL,
                model=ANALYSIS_AGENT_MODEL,
                agent_logger=agent_logger,
                environment_logics=environment_logics
            )
            new_environment_logics_count = len(new_environment_logics.split("### Analysis Result")) - 1
            if environment_logics == "No Analysis Currently":
                environment_logics = new_environment_logics
                cur_new_environment_logics = new_environment_logics
            else:
                cur_num = len(environment_logics.split("### Analysis Result")) - 1
                new_environment_logics = new_environment_logics.strip().split("\n")
                for line in new_environment_logics:
                    if line.strip() == "": continue
                    if line.startswith("### Analysis Result"):
                        cur_num += 1
                        line = f"### Analysis Result {cur_num}:"
                        environment_logics += "\n\n" + line
                        cur_new_environment_logics += "\n\n" + line
                    else:
                        environment_logics += "\n" + line.strip()
                        cur_new_environment_logics += "\n" + line.strip()
            cur_new_environment_logics = cur_new_environment_logics.strip()
            if environment_logics is None:
                print("No environment logics found.")
                break
            with open(f"{base_dir}/turn_{turn}/environment_logics.txt", "w") as f:
                f.write(environment_logics)
        else:
            raise Exception()
            with open(f"{base_dir}/turn_{turn}/environment_logics.txt", "r") as f:
                environment_logics = f.read()
        
        cur_env_rule = optimization_agent.optimize_patch(
            cur_env_rule=cur_env_rule,
            # model=MAIN_MODEL,
            model_code=OPTIMIZATION_AGENT_MODEL_CODE,
            model_valid=OPTIMIZATION_AGENT_MODEL_VALID,
            agent_logger=agent_logger,
            last_environment_logics=last_environment_logics,
            new_environment_logics=cur_new_environment_logics,
        )

        with open(f"alfworld/{initial_interface_module_name}_{EXPERIMENT_NAME}_{date_time}_turn_{turn+1}.py", "w") as f:
            f.write(cur_env_rule)
        interface_module_name = f"{initial_interface_module_name}_{EXPERIMENT_NAME}_{date_time}_turn_{turn+1}"
except Exception as e:
    print(e)
    exc_type, exc_value, exc_traceback = sys.exc_info()
    error_details = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    print(error_details)
finally:
    for turn in range(20):
        system_file = f"alfworld/{initial_interface_module_name}_{EXPERIMENT_NAME}_{date_time}_turn_{turn}.py"
        if os.path.exists(system_file):
            path = f"{base_dir}/turn_{turn}/{initial_interface_module_name}_{EXPERIMENT_NAME}_{date_time}_turn_{turn}.py"
            if not os.path.exists(f"{base_dir}/turn_{turn}"):
                os.makedirs(f"{base_dir}/turn_{turn}", exist_ok=True)
            shutil.move(system_file, path)
            print(f"Moved {system_file} to {path}")
    with open(f"{base_dir}/golden_action_obs.json", "r", encoding="utf-8") as f:
        golden_action_obs = json.load(f)
    with open(f"alfworld/golden_action_obs.json", "w", encoding="utf-8") as f:
        json.dump(golden_action_obs, f, indent=2)

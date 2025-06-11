from experiment_vanilla import SingleAlfredTWEnv
import yaml
import json
import os
AGENTIC_SYSTEM_DEFAULT_MODEL = os.getenv("AGENTIC_SYSTEM_DEFAULT_MODEL", "Qwen2.5-7B-Instruct")
from call_llm import call_llm
import logging
import io

def check_task_id(task_id: str):
    parts = task_id.split('-')
    # 必须要分成两个部分才能符合 "int-int" 的基本格式
    if len(parts) != 2:
        return False, None, None
    
    try:
        task_type_idx = int(parts[0])
        task_idx = int(parts[1])
    except ValueError:
        # 如果无法转换为整数，说明不符合要求
        return False, None, None
    
    # 第一个数字必须在 [0, 5] 范围
    if not (0 <= task_type_idx <= 5):
        return False, None, None
    
    # 第二个数字必须在 [0, 19] 范围
    # if not (0 <= task_idx <= 19):
    #     return False, None, None
    
    # 如果都符合，则返回 True 以及解析出的数字
    return True, task_type_idx, task_idx

import ast
def validate_WrapStep_code(env_rule_code: str):
    # 1. 尝试解析为 AST
    try:
        tree = ast.parse(env_rule_code)
    except SyntaxError:
        return False, None

    # 2. 在 AST 中查找函数定义 WrapStep，并检查形参列表
    WrapStep_def = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'WrapStep':
            # 确认形参个数及命名
            if (len(node.args.args) == 5 and
                node.args.args[0].arg == 'env' and
                node.args.args[1].arg == 'init_obs' and
                node.args.args[2].arg == 'task' and
                node.args.args[3].arg == 'agent_action' and
                node.args.args[4].arg == 'logger'):
                WrapStep_def = node
                break

    if not WrapStep_def:
        return False, None

    # 3. 若 AST 检查通过，则尝试执行代码并获取 WrapStep 函数对象
    env_locals = {}
    try:
        code_obj = compile(env_rule_code, '<string>', 'exec')
        exec(code_obj, env_locals)
    except Exception:
        # 如果执行过程中报错，比如引用了未安装的包等，也返回 False
        return False, None

    func = env_locals.get('WrapStep')
    if not callable(func):
        return False, None

    return True, func

def validate_InferRules_code(env_rule_code: str):
    # 1. 尝试解析为 AST
    try:
        tree = ast.parse(env_rule_code)
    except SyntaxError:
        return False, None

    # 2. 在 AST 中查找函数定义 InferRules，并检查形参列表
    InferRules_def = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'InferRules':
            # 确认形参个数及命名
            if (len(node.args.args) == 2 and
                node.args.args[0].arg == 'init_obs' and
                node.args.args[1].arg == 'task'):
                InferRules_def = node
                break

    if not InferRules_def:
        return False, None

    # 3. 若 AST 检查通过，则尝试执行代码并获取 InferRules 函数对象
    env_locals = {}
    try:
        code_obj = compile(env_rule_code, '<string>', 'exec')
        exec(code_obj, env_locals)
    except Exception:
        # 如果执行过程中报错，比如引用了未安装的包等，也返回 False
        return False, None

    func = env_locals.get('InferRules')
    if not callable(func):
        return False, None

    return True, func

class EnvSimulator:
    def __init__(self):
        pass
    
    def init(self, task_id: str, env_rule_code: str | None):
        eval_result, task_type_idx, task_idx = check_task_id(task_id)
        if not eval_result:
            return False, "Invalid task_id: {task_id}. Must be in the format 'int-int' where int1 in [0, 5]."
        self.task_type_idx = task_type_idx
        self.task_idx = task_idx

        if env_rule_code is not None:
            eval_result, WrapStep_func = validate_WrapStep_code(env_rule_code)
            if not eval_result:
                return False, "Invalid env_rule_code: {env_rule_code}. Must contain a function named 'WrapStep' with parameters 'env', 'agent_action' and 'logger'. And the function should be executable."
            
            self.WrapStep = WrapStep_func
            
            eval_result, InferRules_func = validate_InferRules_code(env_rule_code)
            if not eval_result:
                return False, "Invalid env_rule_code: {env_rule_code}. Must contain a function named 'InferRules' with parameters 'init_obs' and 'task'. And the function should be executable."
            self.InferRules = InferRules_func


            self.env_rule_code = env_rule_code
        else:
            self.WrapStep = None
            self.InferRules = None
            self.env_rule_code = None

        with open(f"alfworld/file_names_train.json", "r") as f:
            FILE_NAMES = json.load(f)
        with open("alfworld/alfworld/configs/base_config.yaml", "r") as f:
            alfworld_config = yaml.safe_load(f)
        
        try:
            env = SingleAlfredTWEnv(alfworld_config, FILE_NAMES[self.task_type_idx][self.task_idx], "train")
        except Exception as e:
            return False, f"Error initializing environment: {e}. The task_id may be invalid."
        
        self.env = env.init_env(batch_size=1)

        self.obs, self.info = self.env.reset()
        self.obs = '\n'.join(self.obs[0].split('\n\n')[1:])
        self.task = self.obs.split('\n')[1].strip()
        self.init_obs = self.obs.split('\n')[0].strip()

        self.action_history = []
        self.have_execute_agent_action = False

        self.messages = [
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

{self.InferRules(self.init_obs, self.task)}"""
            },
            {
                "role": "user",
                "content": f"""# Task

{self.obs}

Firstly, you need to give your complete plan in natural language to solve the task. You should respond in one line with the complete plan in **natural language**, with the format:
Plan: First, I will xxx. Then, I will xxx. ..."""
            }
        ]

        self.plan = call_llm(self.messages, model=AGENTIC_SYSTEM_DEFAULT_MODEL, temperature=0.0, max_tokens=1024)

        self.messages.append({"role": "assistant", "content": self.plan})
        self.messages.append({"role": "user", "content": f"""# Task

{self.obs}

Begin by examining the environment or taking any initial steps you find relevant. Remember, provide only one action each time."""})

        self.log_stream = io.StringIO()
        stream_handler = logging.StreamHandler(self.log_stream)
        stream_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        stream_handler.setFormatter(formatter)
        self.simulator_logger = logging.getLogger("simulator_logger")
        for handler in self.simulator_logger.handlers[:]:
            self.simulator_logger.removeHandler(handler)
        self.simulator_logger.setLevel(logging.DEBUG)
        self.simulator_logger.addHandler(stream_handler)
        self.simulator_logger.propagate = False

        log = f"Initializing environment...\n"
        log += f"Observation: {self.obs}\n"
        log += f"Agent Plan: {self.plan}\n"
        log += f"Action history: {self.action_history}"
        return True, log

    def step(self, action: str):
        self.action_history.append(action)
        obs, reward, done, info = self.env.step([action])
        obs, reward, done = obs[0], info['won'][0], done[0]
        log = f"Executing action: {action}\n"
        log += f"Observation: {obs}\n"
        log += f"Reward: {reward}\n"
        log += f"Done: {done}\n"
        log += f"Action history: {self.action_history}"
        self.messages.append({"role": "assistant", "content": action})
        self.messages.append({"role": "user", "content": f"""# Observation from the environment
{obs}

{self.task}

Now you need to give your next action."""})
        return True, log
    
    def cancel_one_step(self):
        if self.have_execute_agent_action:
            return False, "You have simulated the agent's action in past. To ensure the simulation is consistent, you cannot cancel the action in this simulation. Please reset the environment first."
        if len(self.action_history) == 0:
            return False, "No action history to cancel."

        with open(f"alfworld/file_names_train.json", "r") as f:
            FILE_NAMES = json.load(f)
        with open("alfworld/alfworld/configs/base_config.yaml", "r") as f:
            alfworld_config = yaml.safe_load(f)
        env = SingleAlfredTWEnv(alfworld_config, FILE_NAMES[self.task_type_idx][self.task_idx], "train")
        self.env = env.init_env(batch_size=1)

        self.obs, self.info = self.env.reset()
        self.obs = '\n'.join(self.obs[0].split('\n\n')[1:])
        self.task = self.obs.split('\n')[1].strip()
        obs, reward, done = self.obs, 0, False
       
        for i in range(len(self.action_history) - 1):
            obs, reward, done, info = self.env.step([self.action_history[i]])
            obs, reward, done = obs[0], info['won'][0], done[0]

        log = f"Canceling action: {self.action_history[-1]}\n"
        log += f"The action history executed: {self.action_history[:-1]}\n"
        log += f"Observation: {obs}\n"
        log += f"Reward: {reward}\n"
        log += f"Done: {done}\n"
        log += f"Action history: {self.action_history}"
        self.action_history.pop()
        self.messages = self.messages[:-2]
        return True, log
    
    def reset(self):
        with open(f"alfworld/file_names_train.json", "r") as f:
            FILE_NAMES = json.load(f)
        with open("alfworld/alfworld/configs/base_config.yaml", "r") as f:
            alfworld_config = yaml.safe_load(f)
        env = SingleAlfredTWEnv(alfworld_config, FILE_NAMES[self.task_type_idx][self.task_idx], "train")
        self.env = env.init_env(batch_size=1)

        self.obs, self.info = self.env.reset()
        self.obs = '\n'.join(self.obs[0].split('\n\n')[1:])
        self.task = self.obs.split('\n')[1].strip()
        self.init_obs = self.obs.split('\n')[0].strip()
        self.action_history = []
        self.have_execute_agent_action = False
        self.messages = [
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

{self.InferRules(self.init_obs, self.task)}"""
            },
            {
                "role": "user",
                "content": f"""# Task

{self.obs}

Firstly, you need to give your complete plan in natural language to solve the task. You should respond in one line with the complete plan in **natural language**, with the format:
Plan: First, I will xxx. Then, I will xxx. ..."""
            }
        ]
        self.messages.append({"role": "assistant", "content": self.plan})
        self.messages.append({"role": "user", "content": f"""# Task

{self.obs}

Begin by examining the environment or taking any initial steps you find relevant. Remember, provide only one action each time."""})
        
        log = f"Resetting environment...\n"
        log += f"Observation: {self.obs}\n"
        log += f"Task: {self.task}\n"
        log += f"Agent Plan: {self.plan}\n"
        log += f"Action history: {self.action_history}"
        return True, log

    def execute_agent_action(self, agent_action: str):
        if self.WrapStep is None:
            return False, "No WrapStep function provided. This simulator cannot execute agent actions."
        
        try:
            self.log_stream.seek(0)
            self.log_stream.truncate(0)
            obs, reward, done = self.WrapStep(self.env, self.init_obs, self.task, agent_action, self.simulator_logger)
            log_contents = self.log_stream.getvalue()
        except Exception as e:
            return False, f"Error executing agent action: {e}"
        log = f"Executing agent action: {agent_action}\n"
        log += f"Observation: {obs}\n"
        log += f"Reward: {reward}\n"
        log += f"Done: {done}\n"
        log += f"Action history: {self.action_history}"
        if log_contents:
            log += f"\nLog contents when executing `WrapStep`: {log_contents}"
        self.have_execute_agent_action = True
        self.action_history.append(agent_action)

        self.messages.append({"role": "assistant", "content": agent_action})
        self.messages.append({"role": "user", "content": f"""# Observation from the environment
{obs}

{self.task}

Now you need to give your next action."""})
        return True, log
    
    def get_next_agent_action(self):
        agent_action = call_llm(self.messages, model=AGENTIC_SYSTEM_DEFAULT_MODEL, temperature=0.0, max_tokens=1024)
        log = f"Next agent action: {agent_action}\n"
        return True, log
    
    def change_last_action_observation(self, obs: str):
        self.messages[-1]["content"] = f"""# Observation from the environment
{obs}

{self.task}

Now you need to give your next action."""
        log = f"Changed last action observation to: {obs}\n"
        return True, log
    
    def run_task(self, task_id: str, env_rule_code: str):
        done, log = self.init(task_id, env_rule_code)
        if not done:
            return False, log
        
        log = f"========== Task ID: {task_id} ==========\n"
        log += f"Task: {self.obs}\n"

        same_action_count = 0
        same_action = ""

        for i in range(100):
            agent_action = call_llm(self.messages, model=AGENTIC_SYSTEM_DEFAULT_MODEL, temperature=0.0, max_tokens=1024)
            self.messages.append({"role": "assistant", "content": agent_action})
            log += f"Agent Action: {agent_action}\n"

            self.log_stream.seek(0)
            self.log_stream.truncate(0)
            obs, reward, done = self.WrapStep(self.env, self.init_obs, self.task, agent_action, self.simulator_logger)
            log_contents = self.log_stream.getvalue()

            self.have_execute_agent_action = True
            self.action_history.append(agent_action)
            
            log += f"Observation: {obs}\n"
            log += f"Reward: {reward}\n"
            log += f"Done: {done}\n"
            if log_contents:
                log += f"Log contents when executing `WrapStep`: {log_contents}\n"
            log += f"---------------------------------\n"

            self.messages.append({
    "role": "user",
    "content": f"""# Observation from the environment
{obs}

{self.task}

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
        return True, log

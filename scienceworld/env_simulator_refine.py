from scienceworld import ScienceWorldEnv
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
    
    # 第一个数字必须在 [0, 29] 范围
    if not (0 <= task_type_idx <= 29):
        return False, None, None
    
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
            return False, "Invalid task_id: {task_id}. Must be in the format 'int-int' where int1 in [0, 29]."
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
 
        try:
            env = ScienceWorldEnv("", "", envStepLimit=30)
            taskNames = env.get_task_names()
            taskName = taskNames[task_type_idx]
            env.load(taskName, task_idx, "easy", generateGoldPath=True)
        except Exception as e:
            return False, f"Error initializing environment: {e}. The task_id may be invalid."
        
        self.env = env

        self.obs, self.info = self.env.reset()
        self.task = self.env.taskdescription()[18:]
        self.init_obs = self.obs

        self.action_history = []
        self.have_execute_agent_action = False

        self.messages = [
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

{self.InferRules(self.init_obs, self.task)}"""
        },
        {
            "role": "user",
            "content": f"""# Task

{self.init_obs}

{self.task}

Begin by examining the environment or taking any initial steps you find relevant. Remember, provide only one action each time."""
        }
    ]

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
        log += f"Task: {self.task}\n"
        log += f"Observation: {self.obs}\n"
        log += f"Action history: {self.action_history}"
        return True, log

    def step(self, action: str):
        self.action_history.append(action)
        obs, _, done, info = self.env.step(action)
        log = f"Executing action: {action}\n"
        log += f"Observation: {obs}\n"
        log += f"Score: {info['score']}\n"
        log += f"Done: {done}\n"
        log += f"Action history: {self.action_history}"
        self.messages.append({"role": "assistant", "content": action})
        self.messages.append({"role": "user", "content": f"""# Observation from the environment
{obs}

{self.task}

Now you need to give your next action."""})
        return True, log
    
    def reset(self):
        
        env = ScienceWorldEnv("", "", envStepLimit=30)
        taskNames = env.get_task_names()
        taskName = taskNames[self.task_type_idx]
        env.load(taskName, self.task_idx, "easy", generateGoldPath=True)
        self.env = env

        self.obs, self.info = self.env.reset()
        self.task = self.env.taskdescription()[18:]
        self.init_obs = self.obs

        self.action_history = []
        self.have_execute_agent_action = False
        self.messages = [
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

{self.InferRules(self.init_obs, self.task)}"""
        },
        {
            "role": "user",
            "content": f"""# Task

{self.init_obs}

{self.task}

Begin by examining the environment or taking any initial steps you find relevant. Remember, provide only one action each time."""
        }
    ]

        log = f"Resetting environment...\n"
        log += f"Task: {self.task}\n"
        log += f"Observation: {self.obs}\n"
        log += f"Action history: {self.action_history}"
        return True, log

    def execute_agent_action(self, agent_action: str):
        if self.WrapStep is None:
            return False, "No WrapStep function provided. This simulator cannot execute agent actions."
        
        try:
            self.log_stream.seek(0)
            self.log_stream.truncate(0)
            obs, done, score = self.WrapStep(self.env, self.init_obs, self.task, agent_action, self.simulator_logger)
            log_contents = self.log_stream.getvalue()
        except Exception as e:
            return False, f"Error executing agent action: {e}"
        log = f"Executing agent action: {agent_action}\n"
        log += f"Observation: {obs}\n"
        log += f"Score: {score}\n"
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
        agent_action = call_llm(self.messages, model=AGENTIC_SYSTEM_DEFAULT_MODEL, temperature=0.0, max_tokens=200)

        self.messages.append({"role": "assistant", "content": agent_action})
        self.messages.append({"role": "user", "content": f"""Now you need to review your action and refine it if necessary. You should respond with the same action if you think it is correct. If you want to change your action, please provide the new action. You should respond in the following format:
THOUGHT: your thought in one line
FINAL ACTION: your final action"""})
        agent_action = call_llm(self.messages, model=AGENTIC_SYSTEM_DEFAULT_MODEL, temperature=0.0, max_tokens=1024)
        agent_action = agent_action.split("ACTION:")[-1].strip()
        self.messages = self.messages[:-2]

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
        log += f"Task: {self.task}\n"
        log += f"Observation: {self.obs}\n"

        same_action = ""
        same_action_count = 0

        for i in range(50):
            agent_action = call_llm(self.messages, model=AGENTIC_SYSTEM_DEFAULT_MODEL, temperature=0.0, max_tokens=200)
            self.messages.append({"role": "assistant", "content": agent_action})

            self.messages.append({"role": "user", "content": f"""Now you need to review your action and refine it if necessary. You should respond with the same action if you think it is correct. If you want to change your action, please provide the new action. You should respond in the following format:
THOUGHT: your thought in one line
FINAL ACTION: your final action"""})
            agent_action = call_llm(self.messages, model=AGENTIC_SYSTEM_DEFAULT_MODEL, temperature=0.0, max_tokens=1024)
            agent_action = agent_action.split("ACTION:")[-1].strip()
            self.messages = self.messages[:-2]
            self.messages.append({"role": "assistant", "content": agent_action})
            
            log += f"Agent Action: {agent_action}\n"

            self.log_stream.seek(0)
            self.log_stream.truncate(0)
            obs, done, score = self.WrapStep(self.env, self.init_obs, self.task, agent_action, self.simulator_logger)
            log_contents = self.log_stream.getvalue()

            self.have_execute_agent_action = True
            self.action_history.append(agent_action)
            
            log += f"Observation: {obs}\n"
            log += f"Score: {score}\n"
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
            if score < 0:
                done = True

            if agent_action == same_action:
                same_action_count += 1
            else:
                same_action = agent_action
                same_action_count = 0
            if same_action_count > 6:
                done = True

            if done:
                break
        return True, log

from env_webshop import webshopEnv
import yaml
import json
import os
AGENTIC_SYSTEM_DEFAULT_MODEL = os.getenv("AGENTIC_SYSTEM_DEFAULT_MODEL", "Qwen2.5-7B-Instruct")
from call_llm import call_llm
import logging
import io

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
        try:
            task_idx = int(task_id)
            assert task_idx >= 50 and task_idx <= 199
        except Exception as e:
            return False, "Invalid task_id: {task_id}. Must be in the format 'int' where in [50, 199]."
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
            self.env = webshopEnv(f"fixed_{task_idx}")
        except Exception as e:
            return False, f"Error initializing environment: {e}. The task_id may be invalid."
        
        self.obs, reward, done = self.env.step("reset")
        self.init_obs = self.obs
        self.task = self.init_obs.split("Instruction:")[1].split("[Search]")[0].strip()

        self.action_history = []
        self.have_execute_agent_action = False

        self.messages = [
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

{self.InferRules(self.init_obs, self.task)}"""
        },
        {
            "role": "user",
            "content": f"""# Task

{self.obs}

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
        log += f"Observation: {self.obs}\n"
        log += f"Action history: {self.action_history}"
        return True, log

    def reset(self):
        self.env = webshopEnv(f"fixed_{self.task_idx}")

        self.obs, reward, done = self.env.step("reset")
        self.init_obs = self.obs
        self.task = self.init_obs.split("Instruction:")[1].split("[Search]")[0].strip()

        self.action_history = []
        self.have_execute_agent_action = False

        self.messages = [
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

{self.InferRules(self.init_obs, self.task)}"""
        },
        {
            "role": "user",
            "content": f"""# Task

{self.obs}

Begin by examining the environment or taking any initial steps you find relevant. Remember, provide only one action each time."""
        }
    ]

        log = f"Resetting environment...\n"
        log += f"Observation: {self.obs}\n"
        log += f"Task: {self.task}\n"
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
        log += f"Task: {self.obs}\n"

        for i in range(30):
            agent_action = call_llm(self.messages, model=AGENTIC_SYSTEM_DEFAULT_MODEL, temperature=0.0, max_tokens=1024)
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

            if done:
                break
        return True, log

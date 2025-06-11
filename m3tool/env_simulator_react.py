from tasks import Task, task_iterators
import yaml
import json
import os
AGENTIC_SYSTEM_DEFAULT_MODEL = os.getenv("AGENTIC_SYSTEM_DEFAULT_MODEL", "Qwen2.5-7B-Instruct")
from call_llm import call_llm
import logging
import io

TOOL_DESC = ['You have access to the following tools:\n[1] convert_hex_to_ascii: Converts a hexadecimal string to ASCII. Arguments: hex_string (str)\n    Signature: convert_hex_to_ascii(hex_string: str) -> str\n[2] reverse_string: Reverses a string. Arguments: string (str)\n    Signature: reverse_string(string: str) -> str\n[3] caesar_decode: Decodes a string using the Caesar cipher. Arguments: message (str), shift (int)\n    Signature: caesar_decode(message: str, shift: int) -> str\n[4] string_length: Finds the length of a string. Arguments: string (str)\n    Signature: string_length(string: str) -> int\n[5] minimum_value: Finds the minimum value from given arguments. Arguments: *args (variable number of arguments)\n    Signature: minimum_value(*args) -> int/float\n[6] maximum_value: Finds the maximum value from given arguments. Arguments: *args (variable number of arguments)\n    Signature: maximum_value(*args) -> int/float\n', 'You have access to the following tools:\n[1] count_nucleotides: Counts the occurrences of each nucleotide in a DNA sequence. Arguments: dna_sequence (str)\n    Signature: count_nucleotides(dna_sequence: str) -> dict\n[2] transcribe_dna_to_mrna: Transcribes DNA sequence to mRNA. Arguments: dna_sequence (str)\n    Signature: transcribe_dna_to_mrna(dna_sequence: str) -> str\n[3] translate_mrna_to_amino_acid: Translates mRNA sequence to a chain of amino acids. Arguments: mrna_sequence (str)\n    Signature: translate_mrna_to_amino_acid(mrna_sequence: str) -> str\n[4] find_max_nucleotide: Return the nucleotide (str) with the maximum count (int). Arguments: nucleotide_counts in the form of (k1, v1, k2, v2, ..., kn, vn)\n    Signature: find_max_nucleotide(*args) -> (str, int)\n[5] is_valid_dna_sequence: Checks if the DNA sequence is valid. Arguments: dna_sequence (str)\n    Signature: is_valid_dna_sequence(dna_sequence: str) -> bool\n[6] reverse_transcribe_mrna_to_dna: Reverse transcribes mRNA sequence to DNA. Arguments: mrna_sequence (str)\n    Signature: reverse_transcribe_mrna_to_dna(mrna_sequence: str) -> str\n', 'You have access to the following tools:\n[1] convert_currency: Converts the commodity price to local currency. Arguments: base_price (float), conversion_rate (float)\n    Signature: convert_currency(base_price: float, conversion_rate: float) -> float\n[2] calculate_tariff: Calculates the trade tariff based on the converted price. Arguments: price (float), tariff_rate (float, in %)\n    Signature: calculate_tariff(price: float, tariff_rate: float) -> float\n[3] estimate_final_value: Estimates the final trade value including the tariff. Arguments: price (float), tariff (float)\n    Signature: estimate_final_value(price: float, tariff: float) -> float\n[4] calculator: Evaluates the given expression and returns the result. Accepts a calculation expression as input. For example, "2 + (3 * 4)" will return 14.\n    Signature: calculator(expression: str) -> float\n[5] find_minimum: Finds the minimum value among the given arguments. Accepts variable number of float arguments.\n    Signature: find_minimum(*args: float) -> float\n[6] find_maximum: Finds the maximum value among the given arguments. Accepts variable number of float arguments.\n    Signature: find_maximum(*args: float) -> float\n', 'You have access to the following tools:\n[1] find_flights: Finds flights based on source, destination and date. Arguments: from_location (str), to_location (str), date (str) in YYYY-MM-DD format.\nReturns a list of flights, each represented as a dictionary with keys "from_location", "to_location" (destination), "date", and "price".\nExample: [{"from_location": "A", "to_location": "B", "date": "2023-12-25", "price": 450}]\n    Signature: find_flights(destination: str, date: str) -> List[Dict]\n[2] book_hotel: Books a hotel based on location and preferences. Arguments: location (str), *preferences (variable number of str arguments).\nReturns a list of hotels, each represented as a dictionary with keys "location", "preferences", "price_per_night", and "rating".\nExample: [{"location": "A", "preferences": ["wifi", "pool"], "price_per_night": 120, "rating": 4}]\n    Signature: book_hotel(location: str, *preferences: str) -> List[Dict]\n[3] budget_calculator: Calculates the total budget for a trip. Arguments: flight_price (float), hotel_price_per_night (float), num_nights (int).\nReturns the total budget (float).\n    Signature: budget_calculator(flight_price: float, hotel_price_per_night: float, num_nights: int) -> float\n[4] max: Finds the maximum value among the given arguments. Accepts variable number of float arguments.\n    Signature: max(*args: float) -> float\n[5] min: Finds the minimum value among the given arguments. Accepts variable number of float arguments.\n    Signature: min(*args: float) -> float\n[6] sum: Sums the given arguments. Accepts variable number of float arguments.\n    Signature: sum(*args: float) -> float\n', 'You have access to the following tools:\n[1] click_url: Clicks on a URL. A clickable URL looks like [Clickable \'<url_argument>\'] in the webpage.\nArguments: url (str).\nReturns the rendered content of the webpage after clicking the URL showing on the current rendered page.\n\n    Signature: click_url(url: str) -> str\n[2] go_to_previous_page: Goes back to the previous page. It has no arguments.\nAfter going back to the previous page, return the rendered content of the webpage.\n    Signature: go_to_previous_page() -> str\n[3] scroll_down: Scrolls down the view. It has no arguments.\nReturns the rendered content of the webpage after scrolling down.\n    Signature: scroll_down() -> str\n[4] scroll_up: Scrolls up the view. It has no arguments.\nReturns the rendered content of the webpage after scrolling up.\n    Signature: scroll_up() -> str\n[5] view: Return the current view in string format of the rendered webpage. It has no arguments.\nReturns the rendered content of the webpage.\nYou should call this when you want to see the rendered content of the current webpage.\n    Signature: view() -> str\n[6] calculator: Evaluates the given expression and returns the result. Accepts a calculation expression as input. For example, "2 + (3 * 4)" will return 14.\n    Signature: calculator(expression: str) -> float\n']

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
    
    # 第一个数字必须在 [0, 4] 范围
    if not (0 <= task_type_idx <= 4):
        return False, None, None
    
    # 第二个数字必须在 [4, 5] 范围
    if not (4 <= task_idx <= 5):
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
                node.args.args[1].arg == 'task_name' and
                node.args.args[2].arg == 'instruction' and
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
                node.args.args[0].arg == 'task_name' and
                node.args.args[1].arg == 'task_type_idx'):
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
            return False, "Invalid task_id: {task_id}. Must be in the format 'int-int' where int1 in [0, 4], int2 in [4, 5]."
        self.task_type_idx = task_type_idx
        self.task_idx = task_idx

        if env_rule_code is not None:
            eval_result, WrapStep_func = validate_WrapStep_code(env_rule_code)
            if not eval_result:
                return False, "Invalid env_rule_code: {env_rule_code}. Must contain a function named 'WrapStep' with parameters 'env', 'task_name', 'instruction', 'agent_action', 'logger'. And the function should be executable."
            
            self.WrapStep = WrapStep_func
            
            eval_result, InferRules_func = validate_InferRules_code(env_rule_code)
            if not eval_result:
                return False, "Invalid env_rule_code: {env_rule_code}. Must contain a function named 'InferRules' with parameters 'task_name' and 'task_type_idx'. And the function should be executable."
            self.InferRules = InferRules_func

            self.env_rule_code = env_rule_code
        else:
            self.WrapStep = None
            self.InferRules = None
            self.env_rule_code = None

        self.env: Task = task_iterators[task_type_idx][0][task_idx]
        self.env.reset()

        self.task_name = self.env.name.strip()
        self.instruction = self.env.instruction.strip()

        self.action_history = []
        self.have_execute_agent_action = False

        self.messages = [
            {
                "role": "system",
                "content": f"""You are an AI assistant solving tasks using tools. You have access to the following tools:

{TOOL_DESC[task_type_idx]}

You can use the tools by outputing the tool name followed by its arguments, delimited by commas.
You should begin your tool invocation with 'Action:' and end it with 'End Action'.
Example: 'Action: tool_name, argument_1 End Action'
You can only invoke one tool at a time.

# Environment Rule

{self.InferRules(self.task_name, task_type_idx)}

Remember, provide only one action each time."""
            },
            {
                "role": "user",
                "content": f"""Instruction: {self.instruction}

If you need to output the answer, you should only respond in following format:
Thought: <your thought>
Answer: <your answer>

If you need to call tool, you can only invoke one tool at a time, you should only respond in following format:
Thought: <your thought>
Action: <your action to call tool_name> End Action
"""
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
        log += f"Task: {self.instruction}\n"
        log += f"Tools Description: {TOOL_DESC[task_type_idx]}\n"
        log += f"Action history: {self.action_history}"
        return True, log
    
    def reset(self):
        self.env.free_resource()
        self.env.reset()

        self.task_name = self.env.name.strip()
        self.instruction = self.env.instruction.strip()

        self.action_history = []
        self.have_execute_agent_action = False

        self.messages = [
            {
                "role": "system",
                "content": f"""You are an AI assistant solving tasks using tools. You have access to the following tools:

{TOOL_DESC[self.task_type_idx]}

You can use the tools by outputing the tool name followed by its arguments, delimited by commas.
You should begin your tool invocation with 'Action:' and end it with 'End Action'.
Example: 'Action: tool_name, argument_1 End Action'
You can only invoke one tool at a time.

# Environment Rule

{self.InferRules(self.task_name, self.task_type_idx)}

Remember, provide only one action each time."""
            },
            {
                "role": "user",
                "content": f"""Instruction: {self.instruction}

If you need to output the answer, you should only respond in following format:
Thought: <your thought>
Answer: <your answer>

If you need to call tool, you can only invoke one tool at a time, you should only respond in following format:
Thought: <your thought>
Action: <your action to call tool_name> End Action
"""
            }
        ]
        log = f"Resetting environment...\n"
        log += f"Task: {self.instruction}\n"
        log += f"Action history: {self.action_history}"
        return True, log

    def execute_agent_action(self, agent_action: str):
        if self.WrapStep is None:
            return False, "No WrapStep function provided. This simulator cannot execute agent actions."
        
        try:
            self.log_stream.seek(0)
            self.log_stream.truncate(0)
            obs, reward, done = self.WrapStep(self.env, self.task_name, self.instruction, agent_action, self.simulator_logger)
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
        self.messages.append({"role": "user", "content": f"""{obs}

If you need to output the answer, you should only respond in following format:
Thought: <your thought>
Answer: <your answer>

If you need to call tool, you can only invoke one tool at a time, you should only respond in following format:
Thought: <your thought>
Action: <your action to call tool_name> End Action"""})
        return True, log
    
    def get_next_agent_action(self):
        agent_action = call_llm(self.messages, model=AGENTIC_SYSTEM_DEFAULT_MODEL, temperature=0.0, max_tokens=1024)
        log = f"Agent Output: {agent_action}\n"
        return True, log
    
    def change_last_action_observation(self, obs: str):
        self.messages[-1]["content"] = f"""{obs}

If you need to output the answer, you should only respond in following format:
Thought: <your thought>
Answer: <your answer>

If you need to call tool, you can only invoke one tool at a time, you should only respond in following format:
Thought: <your thought>
Action: <your action to call tool_name> End Action"""
        log = f"Changed last action observation to: {obs}\n"
        return True, log
    
    def run_task(self, task_id: str, env_rule_code: str):
        done, log = self.init(task_id, env_rule_code)
        if not done:
            return False, log
        
        log = f"========== Task ID: {task_id} ==========\n"
        log += f"Task: {self.instruction}\n"

        for i in range(10):
            agent_action = call_llm(self.messages, model=AGENTIC_SYSTEM_DEFAULT_MODEL, temperature=0.0, max_tokens=1024)
            self.messages.append({"role": "assistant", "content": agent_action})
            log += f"Agent Output: {agent_action}\n"
            try:
                if "Action" in agent_action:
                    agent_action = ("Action:" + agent_action.split("Action:")[1]).strip()
                elif "Answer" in agent_action:
                    agent_action = ("Answer:" + agent_action.split("Answer:")[1]).strip()
            except Exception as e:
                pass
            log += f"Agent Action: {agent_action}\n"

            self.log_stream.seek(0)
            self.log_stream.truncate(0)
            obs, reward, done = self.WrapStep(self.env, self.task_name, self.instruction, agent_action, self.simulator_logger)
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
    "content": f"""{obs}

If you need to output the answer, you should only respond in following format:
Thought: <your thought>
Answer: <your answer>

If you need to call tool, you can only invoke one tool at a time, you should only respond in following format:
Thought: <your thought>
Action: <your action to call tool_name> End Action"""})

            if done:
                break
        return True, log

    def __del__(self):
        try:
            self.env.free_resource()
        except Exception as e:
            print(f"Error freeing resources: {e}")

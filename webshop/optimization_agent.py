import json
import sys
# sys.path.append('.')
import os
TEMPLATE = os.getenv("TEMPLATE", "vanilla")
import traceback

import yaml
from call_llm import call_llm
from tqdm import tqdm

if TEMPLATE == "vanilla":
    import optimization_agent_prompt_vanilla as optimization_agent_prompt_vanilla
    from experiment_vanilla import run_single_task
    from env_simulator_vanilla import EnvSimulator
    from env_simulator_vanilla import validate_InferRules_code, validate_WrapStep_code
elif TEMPLATE == "react":
    import optimization_agent_prompt_react as optimization_agent_prompt_vanilla
    from experiment_react import run_single_task
    from env_simulator_react import EnvSimulator
    from env_simulator_react import validate_InferRules_code, validate_WrapStep_code
elif TEMPLATE == "sc":
    import optimization_agent_prompt_sc as optimization_agent_prompt_vanilla
    from experiment_sc import run_single_task
    from env_simulator_sc import EnvSimulator
    from env_simulator_sc import validate_InferRules_code, validate_WrapStep_code
elif TEMPLATE == "refine":
    import optimization_agent_prompt_refine as optimization_agent_prompt_vanilla
    from experiment_refine import run_single_task
    from env_simulator_refine import EnvSimulator
    from env_simulator_refine import validate_InferRules_code, validate_WrapStep_code
elif TEMPLATE == "planning":
    import optimization_agent_prompt_planning as optimization_agent_prompt_vanilla
    from experiment_planning import run_single_task
    from env_simulator_planning import EnvSimulator
    from env_simulator_planning import validate_InferRules_code, validate_WrapStep_code
else:
    raise ValueError(f"Unknown TEMPLATE: {TEMPLATE}")

import re

def process_llm_response(response: str, simulator: EnvSimulator, cur_env_rule):
    # Extract contents of various tags
    action_pattern = r"<action>(.*?)</action>"
    if_need_refine_pattern = r"<if_need_refine>(.*?)</if_need_refine>"
    refine_strategy_pattern = r"<refine_strategy>(.*?)</refine_strategy>"

    # Search for matches
    action_match = re.search(action_pattern, response, flags=re.DOTALL)
    if_need_refine_match = re.search(if_need_refine_pattern, response, flags=re.DOTALL)
    refine_strategy_match = re.search(refine_strategy_pattern, response, flags=re.DOTALL)
    
    # Store found content (if any) – you could log or display these as desired
    if_need_refine = if_need_refine_match.group(1).strip() if if_need_refine_match else ""
    refine_strategy = refine_strategy_match.group(1).strip() if refine_strategy_match else ""

    # If a <conclusion> tag is present, we consider the process finished
    if if_need_refine:
        return True, True, (if_need_refine, refine_strategy)

    # If we have an <action> tag, parse it to decide which simulator method to invoke
    if action_match:
        action_content = action_match.group(1).strip()

        if action_content.startswith("init_simulator(task_id="):
            task_id_match = re.search(r'init_simulator\(task_id\s*=\s*"([^"]+)"\)', action_content)
            if task_id_match:
                task_id_value = task_id_match.group(1)
                success, env_log = simulator.init(task_id_value, cur_env_rule)
                return False, success, env_log
            else:
                return False, False, "[Error]: Invalid Action Format for init_simulator. Expected format: init_simulator(task_id=\"x-y\")"

        elif action_content.startswith("reset_simulator()"):
            success, env_log = simulator.reset()
            return False, success, env_log

        elif action_content.startswith("cancel_one_step()"):
            success, env_log = simulator.cancel_one_step()
            return False, success, env_log

        elif action_content.startswith("step("):
            step_match = re.search(r'step\(action\s*=\s*"([^"]+)"\)', action_content)
            if step_match:
                step_action = step_match.group(1)
                success, env_log = simulator.step(step_action)
                return False, success, env_log
            else:
                return False, False, "[Error]: Invalid Action Format for step. Expected format: step(\"action=xxx\")"

        elif action_content.startswith("execute_agent_action("):
            exec_match = re.search(r'execute_agent_action\(agent_action\s*=\s*"([^"]+)"\)', action_content)
            if exec_match:
                exec_action = exec_match.group(1)
                success, env_log = simulator.execute_agent_action(exec_action)
                return False, success, env_log
            else:
                return False, False, "[Error]: Invalid Action Format for execute_agent_action. Expected format: execute_agent_action(agent_action=\"xxx\")"
        
        elif action_content.startswith("change_last_action_observation("):
            change_match = re.search(r'change_last_action_observation\(obs\s*=\s*"([^"]+)"\)', action_content)
            if change_match:
                change_obs = change_match.group(1)
                success, env_log = simulator.change_last_action_observation(change_obs)
                return False, success, env_log
            else:
                return False, False, "[Error]: Invalid Action Format for change_last_action_observation. Expected format: change_last_action_observation(obs=\"xxx\")"

        elif action_content.startswith("get_next_agent_action()"):
            success, env_log = simulator.get_next_agent_action()
            return False, success, env_log

        elif action_content.startswith("run_task(task_id="):
            task_id_match = re.search(r'run_task\(task_id\s*=\s*"([^"]+)"\)', action_content)
            if task_id_match:
                task_id_value = task_id_match.group(1)
                success, env_log = simulator.run_task(task_id_value, cur_env_rule)
                return False, success, env_log
            else:
                return False, False, "[Error]: Invalid Action Format for run_task. Expected format: run_task(task_id=\"x-y\")"

        else:
            # Unrecognized action format
            return False, False, f"[Error]: Unrecognized action format: {action_content}"

    return False, False, "[Error]: Unrecognized response format. No action or conclusion found."

class OptimizationAgent:
    def __init__(self):
        pass

    def optimize_patch(self, cur_env_rule, model_code, model_valid, agent_logger, last_environment_logics, new_environment_logics):
        init_cur_env_rule = cur_env_rule

        messages = [
            {
                "role": "user",
                "content": optimization_agent_prompt_vanilla.get_optimize_user_prompt(cur_env_rule, last_environment_logics, new_environment_logics)
            }
        ]
        agent_logger.info(f"[OptimizationAgent] messages: {messages}")
        max_tries = 10

        tries = 0
        first_gen_tries = 0
        while True:
            if tries >= max_tries or first_gen_tries >= max_tries:
                agent_logger.warning(f"[OptimizationAgent] Max tries reached. Exiting...")
                cur_env_rule = init_cur_env_rule
                break
            first_gen_tries += 1

            response = call_llm(messages, model=model_code, temperature=0.1, max_tokens=12800)
            messages.append({"role": "assistant", "content": response})
            agent_logger.info(f"[OptimizationAgent] response: {response}")

            if "<code>" not in response or "</code>" not in response:
                if "```python" in response and "```" in response.split("```python")[1]:
                    pass
                else:
                    continue_gen_time = 5
                    while continue_gen_time > 0:
                        continue_gen_time -= 1
                        if "<code>" in response and "</code>" not in response:
                            agent_logger.info(f"[OptimizationAgent] continue_gen_time: {5-continue_gen_time}")
                            print(f"[OptimizationAgent] continue_gen_time: {5-continue_gen_time}")
                            new_messages = messages.copy()
                            final_line_code = response.strip().split("\n")[-1]
                            new_messages.append({"role": "user", "content": f"Continue generating from the last line of code. You should generate '{final_line_code}' firstly, and then continue generating. Do not output anything else! Just output the code and end with </code>."})
                            new_response = call_llm(new_messages, model=model_code, temperature=0.1, max_tokens=12800)
                            agent_logger.info(f"[OptimizationAgent] new_response(continue): {new_response}")
                            if final_line_code.strip() not in new_response:
                                agent_logger.info(f"[OptimizationAgent] Final line code not found in new response. Retrying...")
                                continue
                            response = response.strip()
                            space_number = 0
                            for i in final_line_code:
                                if i == " ": 
                                    space_number += 1
                                else:
                                    break
                            new_response_first_line_code = new_response.strip().split("\n")[0]
                            while final_line_code.strip() not in new_response_first_line_code:
                                new_response = "\n".join(new_response.split("\n")[1:])
                                new_response_first_line_code = new_response.strip().split("\n")[0]
                            response = "\n".join(response.split("\n")[:-1]) + "\n" + " " * space_number + new_response_first_line_code + "\n" + "\n".join(new_response.strip().split("\n")[1:])
                            messages[-1]["content"] = response
                            agent_logger.info(f"[OptimizationAgent] merged response: {response}")
                        else:
                            break

            if ("<code>" in response and "</code>" in response) or ("```python" in response and "```" in response.split("```python")[1]):
                if "<code>" in response and "</code>" in response:
                    cur_env_rule = response.split("<code>")[1].split("</code>")[0]
                else:
                    cur_env_rule = response.split("```python")[1].split("```")[0]
                cur_env_rule = cur_env_rule.strip()
                if cur_env_rule.startswith("```python"):
                    cur_env_rule = cur_env_rule.split("```python")[1].strip()
                elif cur_env_rule.startswith("```"):
                    cur_env_rule = cur_env_rule.split("```")[1].strip()
                if cur_env_rule.endswith("```"):
                    cur_env_rule = cur_env_rule.split("```")[0].strip()
                cur_env_rule = cur_env_rule.strip()

                eval_result, WrapStep = validate_WrapStep_code(cur_env_rule)
                if not eval_result:
                    agent_logger.info(f"[OptimizationAgent] Validate WrapStep function Failed")
                    agent_logger.info(f"[OptimizationAgent] Retrying...")
                    # messages.append({"role": "user", "content": f"Code WrapStep function validation failed. Please revise the code. You should output in the same format as before."})
                    messages.pop(-1)
                    continue
                
                eval_result, InferRules = validate_InferRules_code(cur_env_rule)
                if not eval_result:
                    agent_logger.info(f"[OptimizationAgent] Validate InferRules function Failed")
                    agent_logger.info(f"[OptimizationAgent] Retrying...")
                    # messages.append({"role": "user", "content": f"Code InferRules function validation failed. Please revise the code. You should output in the same format as before."})
                    messages.pop(-1)
                    continue

                try:
                    # run_experiment(task_type_idx=1, split="train", interface_module_name=func, logger=None, _slice=1)
                    run_single_task("train", None, (50, "train"), InferRules, WrapStep)
                    agent_logger.info(f"[OptimizationAgent] Code executed successfully.")
                except Exception as e:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    error_details = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
                    agent_logger.info(f"[OptimizationAgent] Exception: {error_details}")
                    # messages.append({"role": "user", "content": f"Code execution failed. Please revise the code. Error details: {error_details}\n\nYou should output in the same format as before."})
                    messages.pop(-1)
                    continue
                
                messages = [messages[0], messages[-1], {"role": "user", "content": optimization_agent_prompt_vanilla.get_simulate_env_user_prompt()}]
                agent_logger.info(f"[OptimizationAgent] New messages: {messages}")
                simulator = EnvSimulator()
                first_gen_tries = 0
                tries += 1
                MAX_SIMULATE_STEP = 30
                simulate_step = 0
                STOP = False
                while True:
                    response = call_llm(messages, model=model_valid, temperature=0.1)
                    simulate_step += 1
                    messages.append({"role": "assistant", "content": response})
                    agent_logger.info(f"[OptimizationAgent] response: {response}")
                    finished, p1, p2 = process_llm_response(response, simulator, cur_env_rule)
                    if finished:
                        break
                    messages.append({"role": "user", "content": p2})
                    if simulate_step > MAX_SIMULATE_STEP:
                        agent_logger.info(f"[OptimizationAgent] Simulation step limit exceeded. Retrying...")
                        STOP = True
                        break
                    if simulate_step == MAX_SIMULATE_STEP:
                        messages[-1]["content"] += """Now you must give your conclusion, provide it in this format:

<thought> Your reasoning here </thought>
<if_need_refine> True/False </if_need_refine>
<refine_strategy> Your strategy for refining the WrapStep function, if if_need_refine is True </refine_strategy>"""
                        agent_logger.info(f"[OptimizationAgent] add user message: {messages[-1]['content']}")
                    agent_logger.info(f"[OptimizationAgent] add user message: {p2}")
                if STOP:
                    continue
                if p2[0].strip() == "True":
                    messages = [messages[0], messages[1], {"role": "user", "content": f"After simulation and validation, you need to refine the code. Refine strategy: {p2[1].strip()}. You should output in the same format as before."}]
                    agent_logger.info(f"[OptimizationAgent] New messages: {messages}")
                    continue
                elif p2[0].strip() == "False":
                    agent_logger.info(f"[OptimizationAgent] Simulation completed successfully.")
                    agent_logger.info(f"[OptimizationAgent] New environment rule: {cur_env_rule}")
                    break
            else:
                agent_logger.info(f"[OptimizationAgent] Invalid response format. Retrying...")
                # messages.append({"role": "user", "content": "Invalid response format. You must provide the output strictly in the following format:\n<thought>YOUR_THOUGHT_PROCESS_HERE</thought>\n<code>YOUR_CODE_HERE</code>"})
                messages.pop(-1)
        if cur_env_rule == init_cur_env_rule:
            agent_logger.info(f"[OptimizationAgent] No changes made to the environment rule. Exiting...")
        return cur_env_rule

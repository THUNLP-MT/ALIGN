import json
import sys
# sys.path.append('.')
from call_llm import call_llm
from tqdm import tqdm
import os
TEMPLATE = os.getenv("TEMPLATE", "vanilla")

if TEMPLATE == "vanilla":
    import analysis_agent_prompt_vanilla as analysis_agent_prompt_vanilla
    from env_simulator_vanilla import EnvSimulator
elif TEMPLATE == "react":
    import analysis_agent_prompt_react as analysis_agent_prompt_vanilla
    from env_simulator_react import EnvSimulator
elif TEMPLATE == "sc":
    import analysis_agent_prompt_sc as analysis_agent_prompt_vanilla
    from env_simulator_sc import EnvSimulator
elif TEMPLATE == "refine":
    import analysis_agent_prompt_refine as analysis_agent_prompt_vanilla
    from env_simulator_refine import EnvSimulator
elif TEMPLATE == "planning":
    import analysis_agent_prompt_planning as analysis_agent_prompt_vanilla
    from env_simulator_planning import EnvSimulator
else:
    raise ValueError(f"Unknown TEMPLATE: {TEMPLATE}")

import re

def process_llm_response(response: str, simulator: EnvSimulator, cur_env_rule):
    # Extract contents of various tags
    action_pattern = r"<action>(.*?)</action>"
    environment_logic_pattern = r"<environment_logic_and_misalignments>(.*?)</environment_logic_and_misalignments>"

    # Search for matches
    action_match = re.search(action_pattern, response, flags=re.DOTALL)
    environment_logic_match = re.search(environment_logic_pattern, response, flags=re.DOTALL)
    
    # Store found content (if any) – you could log or display these as desired
    environment_logic = environment_logic_match.group(1).strip() if environment_logic_match else ""

    # If a <conclusion> tag is present, we consider the process finished
    if environment_logic:
        return True, True, (environment_logic, )

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

        else:
            # Unrecognized action format
            return False, False, f"[Error]: Unrecognized action format: {action_content}"

    return False, False, "[Error]: Unrecognized response format. No action or conclusion found."

class AnalysisAgent:
    def __init__(self):
        pass
    
    def analyze_logging(self, cur_env_rule, env_logging, model, agent_logger, environment_logics):
        """
        cur_env_rule: str = def WrapStep...
        env_logging: [
            {
                "score": int,
                "logging": str
            }
        ]
        """
        for env_log in tqdm(env_logging):
            if env_log["score"] == 1:
                continue
            
            gen_done = False
            for _ in range(10):
                messages = [
                    {
                        "role": "user",
                        "content": analysis_agent_prompt_vanilla.get_analyze_logging_user_prompt(cur_env_rule, env_log["logging"], environment_logics, "\n".join(env_log["gold_action_obs_sequence"]))
                    }
                ]
                task_id = env_log["logging"].split("Task ID: ")[1].split(" ==========")[0].strip()
                agent_logger.info(f"[AnalysisAgent] messages: {messages}")
                response = call_llm(messages, model=model, temperature=0.1)
                messages.append({"role": "assistant", "content": response})
                agent_logger.info(f"[AnalysisAgent] response: {response}")
                if "<analysis_result>" not in response or "</analysis_result>" not in response:
                    agent_logger.info("[AnalysisAgent] response format error")
                    continue
                analysis_result = response.split("<analysis_result>")[1].split("</analysis_result>")[0].strip()
                if analysis_result != "No Misalignment" and analysis_result != "Found Misalignment":
                    agent_logger.info("[AnalysisAgent] analysis result error")
                    continue
                
                if analysis_result == "Found Misalignment":
                    if "<environment_logic_and_misalignments>" not in response or "</environment_logic_and_misalignments>" not in response:
                        agent_logger.info("[AnalysisAgent] response format error")
                        continue
                
                gen_done = True
                break
            if not gen_done:
                agent_logger.info("[AnalysisAgent] generation failed")
                continue

            if analysis_result == "No Misalignment":
                continue
            else:
                messages.append({"role": "user", "content": analysis_agent_prompt_vanilla.get_simulate_env_user_prompt()})
                agent_logger.info(f"[AnalysisAgent] add user message: {messages[-1]['content']}")
                simulator = EnvSimulator()
                MAX_SIMULATE_STEP = 30
                simulate_step = 0
                STOP = False
                while True:
                    response = call_llm(messages, model=model, temperature=0.1)
                    simulate_step += 1
                    messages.append({"role": "assistant", "content": response})
                    agent_logger.info(f"[AnalysisAgent] response: {response}")
                    if "No Misalignment" in response:
                        STOP = True
                        break
                    finished, p1, p2 = process_llm_response(response, simulator, cur_env_rule)
                    if finished:
                        break
                    messages.append({"role": "user", "content": p2})
                    agent_logger.info(f"[AnalysisAgent] add user message: {p2}")
                    if simulate_step > MAX_SIMULATE_STEP:
                        agent_logger.info("[AnalysisAgent] simulation step exceed limit")
                        STOP = True
                        break
                    if simulate_step == MAX_SIMULATE_STEP:
                        messages[-1]["content"] += """Now you must give your conclusion, provide it in this format:

<thought> Your reasoning here </thought>
<environment_logic_and_misalignments> the new environment rules and misalignments identified by you, which have not been fixed by current `WrapStep` function. </environment_logic_and_misalignments>"""
                        agent_logger.info(f"[AnalysisAgent] add user message: {messages[-1]['content']}")
                if STOP:
                    continue

                environment_logics = p2[0].strip()
                agent_logger.info(f"[AnalysisAgent] analyses: {environment_logics}")
                return environment_logics
        return None

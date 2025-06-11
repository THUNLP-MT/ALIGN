analyze_logging_user_prompt_template = """In modern benchmarks evaluating LLM Agent reasoning capabilities, human designers create an Environment with a set of rules defining how tasks are accomplished. These rules, referred to as the Environment’s World Model, specify the sequence of actions required to achieve specific outcomes.

Meanwhile, the Agent operates based on its own World Model, which it constructs by interpreting the task and environment prompts. The Agent first determines its high-level reasoning intent—its understanding of what needs to be done—and then selects actions according to its internal World Model. However, because the Environment’s World Model is manually crafted and may not be fully conveyed through prompts, the Agent’s World Model might differ, leading to unexpected behavior. For instance, the Agent might choose an action that aligns with its intent but violates the Environment’s rules, or it might misinterpret feedback due to insufficient information from the Environment.

We define a misalignment between the Environment’s World Model and the Agent’s World Model as a situation where:
- The Environment provides feedback that does not sufficiently clarify its World Model, leaving the Agent unable to adjust its understanding of the rules.

Your task is to analyze the logs from a recent task to determine whether such a misalignment occurred, preventing a fair assessment of the Agent’s capabilities. And this misalignment has not been fixed by current `WrapStep` function. Your analysis will guide us in addressing this issue moving forward.

-----------------------------------------------------------------------
### Experimental Environment Evaluation Template

```python
env: Task = task_iterators[task_type_idx][0][task_idx]
env.reset()
task_name = env.name.strip()
instruction = env.instruction.strip()
logger.info(f"========== Task Name: {task_name} | Task ID: {task_type_idx}-{task_idx} ==========")
messages = [
        {
        "role": "system",
        "content": f\"\"\"You are an AI assistant solving tasks using tools. You have access to the following tools:

{TOOL_DESC[task_type_idx]}

You can use the tools by outputing the tool name followed by its arguments, delimited by commas.
You should begin your tool invocation with 'Action:' and end it with 'End Action'.
Example: 'Action: tool_name, argument_1 End Action'
You can only invoke one tool at a time.

# Environment Rule

{InferRules(task_name, task_type_idx)}

Remember, provide only one action each time.\"\"\"
        },
        {
        "role": "user",
        "content": f\"\"\Instruction: {instruction}

If you need to output the answer, you should only respond in following format:
Answer: <your answer>

If you need to call tool, you can only invoke one tool at a time, you should only respond in following format:
Action: <your action to call tool_name> End Action\"\"\"
        }
]
logger.info(f"Task: {instruction}")
logger.info(f"Tools Description: {TOOL_DESC[task_type_idx]}")

for i in range(10):
    agent_action = Agent(messages)
    messages.append({"role": "assistant", "content": agent_action})
    logger.info(f"Agent Action: {agent_action}")

    log_stream.seek(0)
    log_stream.truncate(0)
    obs, reward, done = WrapStep(env, agent_action, function_logger)
    log_content = log_stream.getvalue()
    
    logger.info(f"Observation: {obs}")
    logger.info(f"Reward: {reward}")
    logger.info(f"Done: {done}")
    if log_content:
         logger.info(f"Log contents when executing `WrapStep`: {log_content}\n")
    logger.info(f"---------------------------------")

    messages.append({
        "role": "user",
        "content": f\"\"\"{obs}

If you need to output the answer, you should only respond in following format:
Answer: <your answer>

If you need to call tool, you can only invoke one tool at a time, you should only respond in following format:
Action: <your action to call tool_name> End Action\"\"\"
})
    if done:
        break
```

In this template, the function `InferRules` is used to define the environment rules. The function `WrapStep` handles post-processing of the Agent’s actions (e.g. performing pre-checks, returning more detailed feedback, etc.). This function should not interfere with the Agent’s own reasoning. There current implementation is as follows:

```python
{{ WrapStep }}
```

-----------------------------------------------------------------------
### Environment Logs

```txt
{{ logs }}
```

Here, each `Observation` is the feedback returned to the Agent after it executes a tool.

-----------------------------------------------------------------------
### Environment Logics and Misalignment Analyzed in the Previous Steps

{{ environment_logics }}

-----------------------------------------------------------------------
### Your Task

Determine whether, during this task, there was a misalignment between the Environment’s World Model and the Agent’s World Model that hindered a fair assessment of the Agent’s capabilities. Choose exactly one of the following JSON outputs, wrapped in a Python code block:

If there is NO misalignment (i.e., the Agent’s failures stem from its own errors or limitations, not a mismatch with the Environment’s World Model), output:
<analysis_result> No Misalignment </analysis_result>

If there IS a misalignment (i.e., the Environment’s World Model conflicts with the Agent’s World Model), output:
<analysis_result> Found Misalignment </analysis_result>
<environment_logic_and_misalignments> the new environment rules and misalignments identified by you, which have not been fixed by current `WrapStep` function. </environment_logic_and_misalignments>

The format of the environment logic and misalignment is as follows:
```txt
### Analysis Result 1
Analysis Task ID: xxx
Agent Action Tool Type: xxx # The type of tool the Agent attempted to perform, such as "convert_hex_to_ascii", "scroll_down", etc.
Agent Action Tool Case: xxx # The specific tool the Agent attempted to perform.
Agent High-Level Reasoning Intent: xxx # The Agent's high-level reasoning intent, which may be a general description of the action it was trying to perform.
Environment World Model Rule: xxx # The rule from the Environment's World Model that don't align the Agent's World Model.
Sufficient Environment Feedback: xxx # to offer the Agent adequate information to bridge gaps in understanding the environment's world model. such as "The environment should provide 'xxx' feedback when the Agent attempts..."
Type: "Bug of current WrapStep function" or "Need to add new logic in the WrapStep function"

### Analysis Result 2
...
```

Note: You should not generate duplicate misalignment analysis results as the ones already provided in the `Environment Logics and Misalignment Analyzed in the Previous Steps` section.
"""

def get_analyze_logging_user_prompt(WrapStep: str, logs: str, environment_logics: str):
    return analyze_logging_user_prompt_template.replace("{{ WrapStep }}", WrapStep).replace("{{ logs }}", logs).replace("{{ environment_logics }}", str(environment_logics))

simulate_env_user_prompt_template = """Now you should conduct simulation experiments in the simulator to verify that the environment rules you hypothesized and Misalignment you identified truly exists. You must perform sufficient experiments to confirm or refute your suspicion.

Here are the operations you can use:

1. init_simulator(task_id: str)
   - Initializes a new simulator for the specified `task_id`.
   - `task_id` must be in the format 'int-int' where the first int ∈ [0, 4], the second int ∈ [4, 5].
   - The different task types are mapped as follows:
     {
       0: 'message_decoder',
       1: 'cryptobotanists_plant_dna_sequencer',
       2: 'trade_calculator',
       3: 'travel_itinerary_planning',
       4: 'web_browsing',
     }
   - All subsequent operations occur within this initialized simulator.
   - Each task type has its own toolset, you should only use the task type you analyzed in the previous step.

2. reset_simulator()
   - Resets the current simulator to its initial state.

3. execute_agent_action(agent_action: str)
   - Executes an agent action using the `WrapStep` function.

4. change_last_action_observation(obs: str)
   - Updates the last observation returned by the simulator to the specified `obs`.
   - This is useful for simulating the agent’s next action in a different environment feedback context.

5. get_next_agent_action()
   - Retrieves the next action that the real Agent would perform under the current simulation conditions.
   - Note: The Agent’s choice of the next action is based on the current environment state, including the outcomes of any previous `get_next_agent_action()` call, along with the latest observations.

If you believe you have reached a conclusion from your experiments, provide it in this format:

<thought> Your reasoning here </thought>
<environment_logic_and_misalignments> the new environment rules and misalignments identified by you, which have not been fixed by current `WrapStep` function. </environment_logic_and_misalignments>

The format of the environment logic and misalignment is as follows:
```txt
### Analysis Result 1
Analysis Task ID: xxx
Agent Action Tool Type: xxx # The type of tool the Agent attempted to perform, such as "convert_hex_to_ascii", "scroll_down", etc.
Agent Action Tool Case: xxx # The specific tool the Agent attempted to perform.
Agent High-Level Reasoning Intent: xxx # The Agent's high-level reasoning intent, which may be a general description of the action it was trying to perform.
Environment World Model Rule: xxx # The rule from the Environment's World Model that don't align the Agent's World Model.
Sufficient Environment Feedback: xxx # to offer the Agent adequate information to bridge gaps in understanding the environment's world model. such as "The environment should provide 'xxx' feedback when the Agent attempts..."
Type: "Bug of current WrapStep function" or "Need to add new logic in the WrapStep function"

### Analysis Result 2
...
```

If you need to carry out more operations in the simulator, respond in the following format, specifying exactly one operation per turn:

<thought> Your reasoning here, you should consider all hypotheses if the simulation result is not as expected </thought>
<action> The single operation you wish to perform (e.g., init_simulator(task_id="x-y"), execute_agent_action(agent_action="x"), etc.) </action>

Note:
You should verify the correctness of the following, step by step, through your experiments:
1. environment_rules: Use `execute_agent_action` to confirm that the environment rules you hypothesized are indeed correct, and current `WrapStep` function is not sufficient.
2. agent_intent_description: Obtain the Agent’s intended behavior (e.g., via `get_next_agent_action`) and simulate it by using `WrapStep` to confirm whether it aligns with your description.
3. identified_misalignment: Through changing the environment feedback, you can verify whether the misalignment you identified is indeed correct and the environment feedback you hypothesized is indeed sufficient. You can use `WrapStep` to simulate the agent’s action, then use `change_last_action_observation` to change the environment feedback, and finally use `get_next_agent_action` to check whether the agent can correctly identify the next action.
"""

def get_simulate_env_user_prompt():
    return simulate_env_user_prompt_template
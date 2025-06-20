analyze_logging_user_prompt_template = """In modern benchmarks evaluating LLM Agent reasoning capabilities, human designers create an Environment with a set of rules defining how tasks are accomplished. These rules, referred to as the Environment’s World Model, specify the sequence of actions required to achieve specific outcomes. For example, the Environment’s World Model might dictate that certain actions can only be performed after prerequisite actions.

Meanwhile, the Agent operates based on its own World Model, which it constructs by interpreting the task and environment prompts. The Agent first determines its high-level reasoning intent—its understanding of what needs to be done—and then selects actions according to its internal World Model. However, because the Environment’s World Model is manually crafted and may not be fully conveyed through prompts, the Agent’s World Model might differ, leading to unexpected behavior. For instance, the Agent might choose an action that aligns with its intent but violates the Environment’s rules, or it might misinterpret feedback due to insufficient information from the Environment.

We define a misalignment between the Environment’s World Model and the Agent’s World Model as a situation where:
- The Environment provides feedback that does not sufficiently clarify its World Model, leaving the Agent unable to adjust its understanding of the rules.

Your task is to analyze the logs from a recent task to determine whether such a misalignment occurred, preventing a fair assessment of the Agent’s capabilities. And this misalignment has not been fixed by current `WrapStep` function. Your analysis will guide us in addressing this issue moving forward.

-----------------------------------------------------------------------
### Experimental Environment Evaluation Template

```python
logger.info(f"========== Task ID: {task_idx} ==========")
env = webshopEnv(url)
obs, reward, done = env.step("reset")
init_obs = obs
task = init_obs.split("Instruction:")[1].split("[Search]")[0].strip()
messages = [
        {
        "role": "system",
        "content": f\"\"\"You are an AI assistant solving tasks in a webshop environment. Your goal is to help human buy the item they need.

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

{InferRules(init_obs, task)}\"\"\"
        },
        {
        "role": "user",
        "content": f\"\"\"# Task

{obs}

Begin by examining the environment or taking any initial steps you find relevant. Remember, provide only one action each time.\"\"\"
        }
]
logger.info(f"Task: {obs}")

for i in range(30):
    agent_action = Agent(messages)
    messages.append({"role": "assistant", "content": agent_action})
    messages.append({"role": "user", "content": f\"\"\"Now you need to review your action and refine it if necessary. You should respond with the same action if you think it is correct. If you want to change your action, please provide the new action. You should respond in the following format:
THOUGHT: your thought in one line
FINAL ACTION: your final action\"\"\"})
    agent_action = call_llm(messages, model=AGENTIC_SYSTEM_DEFAULT_MODEL, temperature=0.0, max_tokens=1024, llm_port_idx=llm_port_idx)
    agent_action = agent_action.split("ACTION:")[-1].strip()
    messages = messages[:-2]
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
        "content": f\"\"\"# Observation from the environment
{obs}

{task}

Now you need to give your next action.\"\"\"
})
    if done:
        break
```

In this template, the function `InferRules` is used to define the environment rules. The function `WrapStep` handles post-processing of the Agent’s actions. This function should not interfere with the Agent’s own reasoning. There current implementation is as follows:

```python
{{ WrapStep }}
```

-----------------------------------------------------------------------
### Environment Logs

```txt
{{ logs }}
```

Here, each `Observation` is the feedback returned to the Agent after it executes an action.

-----------------------------------------------------------------------
### Environment Logics and Misalignment Analyzed in the Previous Steps

{{ environment_logics }}

-----------------------------------------------------------------------
### Your Task

Determine whether, during this task, there was a misalignment between the Environment’s World Model and the Agent’s World Model that hindered a fair assessment of the Agent’s capabilities.

If there is NO misalignment (i.e., the Agent’s failures stem from its own errors or limitations, not a mismatch with the Environment’s World Model), output:
<analysis_result> No Misalignment </analysis_result>

If there IS a misalignment (i.e., the Environment’s World Model conflicts with the Agent’s World Model), output:
<analysis_result> Found Misalignment </analysis_result>
<environment_logic_and_misalignments> the new environment rules and misalignments identified by you, which have not been fixed by current `WrapStep` function. </environment_logic_and_misalignments>

The format of the environment logic and misalignment is as follows:
```txt
### Analysis Result 1
Analysis Task ID: xxx
Agent Action Type: xxx # The type of action the Agent attempted to perform, such as "search", "click", etc.
Agent Action Case: xxx # The specific action the Agent attempted to perform.
Agent High-Level Reasoning Intent: xxx # The Agent's high-level reasoning intent, which may be a general description of the action it was trying to perform.
Environment World Model Rule: xxx # The rule from the Environment's World Model that don't align the Agent's World Model.
Sufficient Environment Feedback: xxx # to offer the Agent adequate information to bridge gaps in understanding the environment's world model. such as "The environment should provide 'xxx' feedback when the Agent attempts to click on a product without first searching for it."
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
   - `task_id` must be in the format 'int' where the int ∈ [50, 199].
   - All subsequent operations occur within this initialized simulator.

2. reset_simulator()
   - Resets the current simulator to its initial state.

3. execute_agent_action(agent_action: str)
   - Executes an agent action using the `WrapStep` function.

4. change_last_action_observation(obs: str)
   - Updates the last observation returned by the simulator to the specified `obs`.
   - This is useful for simulating the agent’s next action in a different environment feedback context.

5. get_next_agent_action()
   - Retrieves the next action that the real Agent would perform under the current simulation conditions.
   - Note: The Agent’s choice of the next action is based on the current environment state, including the outcomes of any previous `step()` or `get_next_agent_action()` call, along with the latest observations.

If you believe you have reached a conclusion from your experiments, provide it in this format:

<thought> Your reasoning here </thought>
<environment_logic_and_misalignments> the new environment rules and misalignments identified by you, which have not been fixed by current `WrapStep` function. </environment_logic_and_misalignments>

The format of the environment logic and misalignment is as follows:
```txt
### Analysis Result 1
Analysis Task ID: xxx
Agent Action Type: xxx # The type of action the Agent attempted to perform, such as "search", "click", etc.
Agent Action Case: xxx # The specific action the Agent attempted to perform.
Agent High-Level Reasoning Intent: xxx # The Agent's high-level reasoning intent, which may be a general description of the action it was trying to perform.
Environment World Model Rule: xxx # The rule from the Environment's World Model that don't align the Agent's World Model.
Sufficient Environment Feedback: xxx # to offer the Agent adequate information to bridge gaps in understanding the environment's world model. such as "The environment should provide 'xxx' feedback when the Agent attempts to click on a product without first searching for it."
Type: "Bug of current WrapStep function" or "Need to add new logic in the WrapStep function"

### Analysis Result 2
...
```

If you need to carry out more operations in the simulator, respond in the following format, specifying exactly one operation per turn:

<thought> Your reasoning here, you should consider all hypotheses if the simulation result is not as expected </thought>
<action> The single operation you wish to perform (e.g., init_simulator(task_id="x"), execute_agent_action(agent_action="x"), etc.) </action>

Note:
You should verify the correctness of the following, step by step, through your experiments:
1. environment_rules: Use `execute_agent_action` to confirm that the environment rules you hypothesized are indeed correct, and current `WrapStep` function is not sufficient.
2. agent_intent_description: Obtain the Agent’s intended behavior (e.g., via `get_next_agent_action`) and simulate it by using `WrapStep` to confirm whether it aligns with your description.
3. identified_misalignment: Through chaning the environment feedback, you can verify whether the misalignment you identified is indeed correct and the environment feedback you hypothesized is indeed sufficient. You can use `WrapStep` to simulate the agent’s action, then use `change_last_action_observation` to change the environment feedback, and finally use `get_next_agent_action` to check whether the agent can correctly identify the next action.
"""

def get_simulate_env_user_prompt():
    return simulate_env_user_prompt_template
analyze_logging_user_prompt_template = """In modern benchmarks evaluating LLM Agent reasoning capabilities, human designers create an Environment with a set of rules defining how tasks are accomplished. These rules, referred to as the Environment’s World Model, specify the sequence of actions required to achieve specific outcomes. For example, the Environment’s World Model might dictate that certain actions (e.g., operating on a receptacle) can only be performed after prerequisite actions (e.g., moving to the receptacle).

Meanwhile, the Agent operates based on its own World Model, which it constructs by interpreting the task and environment prompts. The Agent first determines its high-level reasoning intent—its understanding of what needs to be done—and then selects actions according to its internal World Model. However, because the Environment’s World Model is manually crafted and may not be fully conveyed through prompts, the Agent’s World Model might differ, leading to unexpected behavior. For instance, the Agent might choose an action that aligns with its intent but violates the Environment’s rules, or it might misinterpret feedback due to insufficient information from the Environment.

We define a misalignment between the Environment’s World Model and the Agent’s World Model as a situation where:
- The Environment provides feedback that does not sufficiently clarify its World Model, leaving the Agent unable to adjust its understanding of the rules.

Your task is to analyze the logs from a recent task to determine whether such a misalignment occurred, preventing a fair assessment of the Agent’s capabilities. And this misalignment has not been fixed by current `WrapStep` function. Your analysis will guide us in addressing this issue moving forward.

-----------------------------------------------------------------------
### Experimental Environment Evaluation Template

```python
logger.info(f"========== Task Type: {task_type[task_type_idx]} | Task ID: {task_type_idx}-{idx} ==========")
env = SingleAlfredTWEnv(alfworld_config, file_name, split)
env = env.init_env(batch_size=1)
obs, info = env.reset()
obs = '\n'.join(obs[0].split('\n\n')[1:])
task = obs.split('\n')[1].strip()
init_obs = obs.split('\n')[0].strip()
messages = [
        {
        "role": "system",
        "content": f\"\"\"You are an AI assistant solving tasks in a household environment. Your goal is to break down complex tasks into simple steps and plan your actions accordingly.

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

for i in range(100):
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
        "content": f\"\"\"# Observation from the environment
{obs}

{task}

Now you need to give your next action.\"\"\"
})
    if done:
        break
```

In this template, the function `InferRules` is used to define the environment rules. The function `WrapStep` handles post-processing of the Agent’s actions (e.g., splitting them into multiple steps, performing pre-checks, returning more detailed feedback, etc.). This function should not interfere with the Agent’s own reasoning. There current implementation is as follows:

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
### Gold Action and Observation Sequence

```txt
{{ gold_action_obs_sequence }}
```

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
Agent Action Type: xxx # The type of action the Agent attempted to perform, such as "examine", "move object to receptacle", etc.
Agent Action Case: xxx # The specific action the Agent attempted to perform.
Agent High-Level Reasoning Intent: xxx # The Agent's high-level reasoning intent, which may be a general description of the action it was trying to perform.
Environment World Model Rule: xxx # The rule from the Environment's World Model that don't align the Agent's World Model.
Sufficient Environment Feedback: xxx # to offer the Agent adequate information to bridge gaps in understanding the environment's world model. such as "The environment should provide 'xxx' feedback when the Agent attempts to operate on a receptacle without first going to it."
Type: "Bug of current WrapStep function" or "Need to add new logic in the WrapStep function"

### Analysis Result 2
...
```

Note: You should not generate duplicate misalignment analysis results as the ones already provided in the `Environment Logics and Misalignment Analyzed in the Previous Steps` section.
"""

def get_analyze_logging_user_prompt(WrapStep: str, logs: str, environment_logics: str, gold_action_obs_sequence: str):
    return analyze_logging_user_prompt_template.replace("{{ WrapStep }}", WrapStep).replace("{{ logs }}", logs).replace("{{ environment_logics }}", str(environment_logics)).replace("{{ gold_action_obs_sequence }}", str(gold_action_obs_sequence))

simulate_env_user_prompt_template = """Now you should conduct simulation experiments in the simulator to verify that the environment rules you hypothesized and Misalignment you identified truly exists. You must perform sufficient experiments to confirm or refute your suspicion.

Here are the operations you can use:

1. init_simulator(task_id: str)
   - Initializes a new simulator for the specified `task_id`.
   - `task_id` must be in the format 'int-int' where the first int ∈ [0, 5].
   - The different task types are mapped as follows:
     {
       0: 'pick_and_place',
       1: 'pick_clean_and_place',
       2: 'pick_heat_and_place',
       3: 'pick_cool_and_place',
       4: 'look_at_or_examine_in_light',
       5: 'pick_two_obj_and_place'
     }
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
Agent Action Type: xxx # The type of action the Agent attempted to perform, such as "examine", "move object to receptacle", etc.
Agent Action Case: xxx # The specific action the Agent attempted to perform.
Agent High-Level Reasoning Intent: xxx # The Agent's high-level reasoning intent, which may be a general description of the action it was trying to perform.
Environment World Model Rule: xxx # The rule from the Environment's World Model that don't align the Agent's World Model.
Sufficient Environment Feedback: xxx # to offer the Agent adequate information to bridge gaps in understanding the environment's world model. such as "The environment should provide 'xxx' feedback when the Agent attempts to operate on a receptacle without first going to it."
Type: "Bug of current WrapStep function" or "Need to add new logic in the WrapStep function"

### Analysis Result 2
...
```

If you need to carry out more operations in the simulator, respond in the following format, specifying exactly one operation per turn:

<thought> Your reasoning here, you should consider all hypotheses if the simulation result is not as expected </thought>
<action> The single operation you wish to perform (e.g., init_simulator(task_id="x-y"), step(action="x"), execute_agent_action(agent_action="x"), etc.) </action>

Note:
You should verify the correctness of the following, step by step, through your experiments:
1. environment_rules: Use `execute_agent_action` to confirm that the environment rules you hypothesized are indeed correct, and current `WrapStep` function is not sufficient.
2. agent_intent_description: Obtain the Agent’s intended behavior (e.g., via `get_next_agent_action`) and simulate it by using `WrapStep` to confirm whether it aligns with your description.
3. identified_misalignment: Through chaning the environment feedback, you can verify whether the misalignment you identified is indeed correct and the environment feedback you hypothesized is indeed sufficient. You can use `WrapStep` to simulate the agent’s action, then use `change_last_action_observation` to change the environment feedback, and finally use `get_next_agent_action` to check whether the agent can correctly identify the next action.
"""

def get_simulate_env_user_prompt():
    return simulate_env_user_prompt_template
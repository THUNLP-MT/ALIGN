optimize_user_prompt_template = """In modern benchmarks evaluating LLM Agent reasoning capabilities, human designers create an Environment with a set of rules defining how tasks are accomplished. These rules, referred to as the Environment’s World Model, specify the sequence of actions required to achieve specific outcomes. For example, the Environment’s World Model might dictate that certain actions (e.g., operating on a receptacle) can only be performed after prerequisite actions (e.g., moving to the receptacle).

Meanwhile, the Agent operates based on its own World Model, which it constructs by interpreting the task and environment prompts. The Agent first determines its high-level reasoning intent—its understanding of what needs to be done—and then selects actions according to its internal World Model. However, because the Environment’s World Model is manually crafted and may not be fully conveyed through prompts, the Agent’s World Model might differ, leading to unexpected behavior. For instance, the Agent might choose an action that aligns with its intent but violates the Environment’s rules, or it might misinterpret feedback due to insufficient information from the Environment.

We define a misalignment between the Environment’s World Model and the Agent’s World Model as a situation where:
- The Environment provides feedback that does not sufficiently clarify its World Model, leaving the Agent unable to adjust its understanding of the rules.

Your task is to refine the environment’s behavior based on the misalignment identified by the AnalysisAgent, ensuring the Agent’s true intentions are executed and its reasoning capabilities are fairly assessed.

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
### Environment Logics and Misalignment Analyzed by AnalysisAgent Previously

{{ last_environment_logics }}

-----------------------------------------------------------------------
### New Environment Logics and Misalignment Analyzed by AnalysisAgent

{{ new_environment_logics }}

-----------------------------------------------------------------------
### Your Task

Based on the misalignments identified by the AnalysisAgent, you need to refine and enhance the `InferRules` function and `WrapStep` function to align the Environment’s World Model with the Agent’s actions and provide clearer feedback. Your output should present the new versions of these functions, ensuring the Agent’s high-level reasoning intent is preserved.
Please ensure you follow these requirements:

1. **Function Signature**  
   The function signature must be:
   ```python
   def InferRules(init_obs, task)
     - init_obs: str, the initial observation from the environment, containing all receptacles.
     - task: str, the task description.

   def WrapStep(env, init_obs, task, agent_action: str, logger)
   ```

2. **Return Values**
   The `InferRules` function’s return value must be a string that describes the environment rules.

   The `WrapStep` function’s return value must be three items:
   ```python
   obs: str, reward: bool, done: bool
   ```

3. **`env.step` Usage**  
   The only permitted usage pattern for `env.step` is:
   ```python
   obs, reward, done, info = env.step([agent_action])
   obs, reward, done = obs[0], info['won'][0], done[0]
   ```
   No alternative usage forms are allowed. Each call to env.step causes an irreversible change to the environment state; actions must therefore be chosen carefully.

4. **Package Imports**  
   You may import other packages if necessary, but you must include all imports in your code.

5. **Multiple Calls & Conditional Returns**  
   You are free to call `env.step` multiple times or return different `obs` depending on `agent_action` or the outcomes of these calls.

6. **You can use logger.debug**
   You can use `logger.debug` to log any information you find useful. The logging will be captured and returned to you in the future for further analysis.

7. Do not modify any aspects not explicitly identified by the AnalysisAgent in the “New Environment Logics and Misalignment Analyzed by AnalysisAgent” section.

8. You must use the following approach when addressing the identified misalignment:
	- For each action defined in environment, provide clear, informative, and sufficient feedback from the environment whenever an invalid action is attempted, guiding the Agent toward understanding and adhering to the environment’s rules.

9. **Output Format**  
   You must provide the output strictly in the following format:
   <thought>YOUR_THOUGHT_PROCESS_HERE</thought>
   <code>YOUR_CODE_HERE</code>

Please ensure your final answer follows these guidelines so that we can accurately bridge the misalignment and allow the environment to execute the Agent’s true intentions.
"""

def get_optimize_user_prompt(WrapStep, last_environment_logics, new_environment_logics):
    return optimize_user_prompt_template.replace("{{ WrapStep }}", WrapStep).replace("{{ last_environment_logics }}", str(last_environment_logics)).replace("{{ new_environment_logics }}", str(new_environment_logics))

simulate_env_user_prompt_template = """Now you should conduct simulation experiments in the simulator to verify if the `InferRules` and `WrapStep` function you provided is correct for the new environment logics and misalignment analyzed by the AnalysisAgent.

You must perform sufficient experiments to confirm or refute your suspicion. Here are the operations you can use:

1. init_simulator(task_id: str)
   - Initializes a new simulator for the specified `task_id`.
   - `task_id` must be in the format 'int-int' where the first int ∈ [0, 5] and the second int ∈ [0, 19].
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
   - Executes an agent action using the `WrapStep` function you generated.

4. change_last_action_observation(obs: str)
   - Updates the last observation returned by the simulator to the specified `obs`.
   - This is useful for simulating the agent’s next action in a different environment feedback context.

5. get_next_agent_action()
   - Retrieves the next action that the real Agent would perform under the current simulation conditions.
   - Note: The Agent’s choice of the next action is based on the current environment state, including the outcomes of any previous `step()` or `get_next_agent_action()` call, along with the latest observations.

6. run_task(task_id: str)
   - Runs the entire task in the simulator and returns the running log.
   - After running the whole task, you need to call `init_simulator` or `reset_simulator` to reinitialize the simulator for further operations.

If you believe you have reached a conclusion from your experiments, provide it in this format:

<thought> Your reasoning here </thought>
<if_need_refine> True/False </if_need_refine>
<refine_strategy> Your strategy for refining the WrapStep function, if if_need_refine is True </refine_strategy>

If you need to carry out more operations in the simulator, respond in the following format, specifying exactly one operation per turn:

<thought> Your reasoning here, you should consider all hypotheses if the simulation result is not as expected </thought>
<action> The single operation you wish to perform (e.g., init_simulator(task_id="x-y"), step(action="x"), execute_agent_action(agent_action="x"), etc.) </action>
"""

def get_simulate_env_user_prompt():
   return simulate_env_user_prompt_template

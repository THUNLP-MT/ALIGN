def InferRules(task_name, task_type_idx):
    """
    Contains the rules for environment and task execute logic for different task types.
    """
    return "There is no rule for this environment."

def WrapStep(env, task_name, instruction, agent_action: str, logger):
    """
    Process the agent action and return the next observation, reward, and done status.
    """
    obs, reward, done = env.step(agent_action)
    return obs, reward, done

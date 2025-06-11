def InferRules(init_obs, task):
    """
    Contains the rules for environment and task execute logic.
    """
    return "There is no rule for this environment."

def WrapStep(env, init_obs, task, agent_action: str, logger):
    """
    Process the agent action and return the next observation, reward, and done status.
    """
    obs, reward, done = env.step(agent_action)
    return obs, reward, done

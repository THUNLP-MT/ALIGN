def InferRules(init_obs, task):
    """
    Contains the rules for environment and task execute logic for different task types.
    """
    return "There is no rule for this environment."

def WrapStep(env, init_obs, task, agent_action: str, logger):
    """
    Process the agent action and return the next observation, done status and score(returned by the environment).
    """
    obs, _, done, info = env.step(agent_action)
    return obs, done, info["score"]
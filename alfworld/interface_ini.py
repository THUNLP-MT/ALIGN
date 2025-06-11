def InferRules(init_obs, task):
    """
    Contains the rules for environment and task execute logic for different task types.
    """
    return "There is no rule for this environment."

def WrapStep(env, init_obs, task, agent_action: str, logger):
    """
    Process the agent action and return the next observation, reward, and done status.
    """
    obs, reward, done, info = env.step([agent_action])
    obs, reward, done = obs[0], info['won'][0], done[0]
    return obs, reward, done

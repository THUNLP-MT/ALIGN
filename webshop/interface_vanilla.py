import re
import logging

# Assuming logger is configured elsewhere in the main script
# Example configuration:
# import sys
# logger = logging.getLogger('EnvironmentWrapper')
# logger.setLevel(logging.DEBUG)
# handler = logging.StreamHandler(sys.stdout)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)

def InferRules(init_obs, task):
    """
    Contains the rules for environment and task execute logic.
    Adds specific rules based on analysis to clarify environment behavior.
    """
    # Rule added based on Analysis Result 1 (Unchanged from previous step)
    buy_rule = """
# Environment Rule Specifics:
- The 'click[Buy]' or 'click[Buy Now]' action can only be successfully executed from the main Item page (the page showing product options, description button, and the buy button).
- Attempting to buy from other pages, such as the Item Description page (reached via 'click[Description]' or 'click[Desc/Overview]'), will result in an error. You must navigate back to the main Item page first (e.g., using 'click[< Prev]') before buying.
"""
    return buy_rule


def WrapStep(env, init_obs: str, task: str, agent_action: str, logger: logging.Logger):
    """
    Process the agent action:
    - Intercepts invalid actions based on known rules (e.g., buying from description page).
    - Provides informative feedback for invalid actions.
    - Executes valid actions using env.step.
    - Returns the next observation, reward, and done status.

    Args:
        env: The environment instance.
        init_obs: The observation *before* the agent took the current action.
        task: The task description.
        agent_action: The action string provided by the agent.
        logger: Logger object for debugging.

    Returns:
        Tuple[str, float, bool]: obs, reward, done
    """
    obs = ""
    reward = 0.0
    done = False

    # Normalize action for easier checking
    normalized_action = agent_action.strip().lower()

    # Check for the specific misalignment: Trying to buy from the description page
    is_buy_action = normalized_action.startswith("click[buy")

    if is_buy_action:
        # Log the full init_obs before performing the state check
        logger.debug(f"Full init_obs before state check for buy action: {init_obs}")

        # Infer state from the observation *before* the action (init_obs)
        # Refined Heuristic: Check for presence of "prev" (likely in '< Prev')
        # and absence of "buy now" in the lowercased observation content.
        lower_init_obs = init_obs.lower()
        # Use core text fragments for flexibility and case-insensitivity
        has_prev_indicator = "prev" in lower_init_obs
        has_buy_now_indicator = "buy now" in lower_init_obs

        is_likely_description_page = has_prev_indicator and not has_buy_now_indicator
        logger.debug(f"Checking for description page state before buy action: has_prev_indicator={has_prev_indicator}, has_buy_now_indicator={has_buy_now_indicator}, is_likely_description_page={is_likely_description_page}")


        if is_likely_description_page:
            logger.debug(f"Intercepted invalid action: '{agent_action}'. Agent attempted to buy from a description page (based on refined check).")
            # Provide specific feedback based on Analysis Result 1
            obs = (
                f"Action '{agent_action}' is invalid in the current state (Description page). "
                "You can only buy from the main item page. "
                "Please go back to the item page first, likely by using an action like 'click[< Prev]'.\n\n"
                f"Previous Observation:\n{init_obs}" # Return the previous observation so the agent knows where it was
            )
            reward = 0.0 # No reward for invalid action
            done = False # Task is not done
            logger.debug(f"Returning custom feedback for invalid buy action. Obs: {obs[:100]}..., Reward: {reward}, Done: {done}")
            return obs, reward, done
        else:
            # Buy action attempted, but not detected as being from description page (presumably valid)
            logger.debug(f"Executing potentially valid buy action: {agent_action} (State check did not indicate description page)")
            obs, reward, done = env.step(agent_action)
            logger.debug(f"Executed env.step for buy action. Obs: {obs[:100]}..., Reward: {reward}, Done: {done}")
            return obs, reward, done
    else:
        # Action is not a buy action, execute normally
        logger.debug(f"Executing non-buy action: {agent_action}")
        obs, reward, done = env.step(agent_action)
        logger.debug(f"Executed env.step for non-buy action. Obs: {obs[:100]}..., Reward: {reward}, Done: {done}")
        return obs, reward, done
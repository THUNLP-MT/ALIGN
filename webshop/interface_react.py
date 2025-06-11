import logging
import re # Potentially useful for parsing actions

# Global variable to store the last search query.
# IMPORTANT: This assumes the calling environment resets this variable to None
# before starting a new task (e.g., after env.step("reset")).
last_search_query = None

def InferRules(init_obs, task):
    """
    Contains the rules for environment and task execute logic.
    Refined based on AnalysisAgent feedback.
    """
    # Refined based on Analysis Result 3, 4, and 5
    return """Environment Rules:
1. Action Format: Use 'Type[Argument]', e.g., 'search[query]', 'click[Product Title]'. Ensure the Argument exactly matches the clickable text or intended search query.
2. Pagination: UI elements like '[Next >]' might be present but not always functional. If a 'click[Next >]' or 'click[Next Page]' action results in an error or no change, it likely means pagination is not supported or applicable. All available results might already be displayed.
3. Search Behavior: Repeating the exact same search query consecutively will yield the exact same results. Consider refining your query or exploring the current results if you are looking for different items.
4. Action Variations: While the action space provides general formats (e.g., 'click[Buy]'), the exact clickable text in the environment might differ slightly (e.g., 'Buy Now'). If an action like 'click[Buy]' fails with an 'Invalid action' message, check the observation for buttons like 'Buy Now' and try the corresponding action (e.g., 'click[Buy Now]'). The environment will provide specific feedback for common cases like this.
5. Item Options: Before you can use 'click[Buy Now]' or similar purchase actions, you MUST select ALL required options for the item (e.g., size, color, style). Check the item page observation carefully for available options and click on them first using 'click[Option Name]'. If a purchase action fails, double-check that every required option has been selected based on the current observation.""" # Updated Rule 5 based on Analysis Result 5

def WrapStep(env, init_obs, task, agent_action: str, logger: logging.Logger):
    """
    Process the agent action, potentially modifying behavior or feedback based on
    AnalysisAgent findings, and return the next observation, reward, and done status.

    Args:
        env: The environment instance.
        init_obs: The initial observation for the task (not typically used here but part of signature).
        task: The task description string (not typically used here but part of signature).
        agent_action: The action string provided by the agent.
        logger: The logger instance.

    Returns:
        A tuple containing:
        - obs (str): The observation resulting from the action.
        - reward (float): The reward obtained.
        - done (bool): Whether the episode has ended.
    """
    global last_search_query
    logger.debug(f"Processing action: {agent_action}")
    logger.debug(f"Last search query before processing: {last_search_query}")

    obs, reward, done = None, 0.0, False
    action_lower = agent_action.lower() # Normalize action for easier comparison

    # --- Analysis Result 1: Handle Pagination Click ---
    if action_lower in ["click[next >]", "click[next page]"]:
        logger.debug(f"Detected pagination action: {agent_action}. Executing env.step.")
        obs, reward, done = env.step(agent_action)
        # Check if the environment indicated an invalid action.
        # Assuming "invalid action" is a key indicator from the environment itself.
        if "invalid action" in obs.lower():
             logger.debug(f"Pagination action resulted in invalid action. Providing specific feedback.")
             # Provide more specific feedback, potentially prepending to the original obs
             obs = "Pagination ('Next >' or 'Next Page') failed or is not supported here. All results might be displayed, or navigation works differently. Please try another action based on the current view.\n\n" + obs
             # If 'invalid action' implies no state change, reset reward/done.
             reward = 0.0
             done = False
        return obs, reward, done

    # --- Analysis Result 3: Handle 'click[Buy]' vs 'click[Buy Now]' ---
    # Intercept 'click[Buy]' as it's explicitly identified as potentially incorrect format.
    elif action_lower == "click[buy]":
        logger.debug(f"Detected 'click[Buy]' action. Providing specific feedback instead of executing.")
        # Do not execute env.step("click[Buy]")
        obs = "Action 'click[Buy]' is ambiguous or incorrect. The action space description uses 'click[Buy]' as a general example, but on the actual item page, you usually need to click a specific button like 'Buy Now'. Please check the observation for the correct button text (e.g., 'Buy Now') and use that in your action (e.g., 'click[Buy Now]')."
        reward = 0.0 # No reward for incorrect action format
        done = False # Episode does not end
        logger.debug(f"Provided specific feedback for 'click[Buy]'. Returned obs: '{obs}'")
        return obs, reward, done

    # --- Analysis Result 2: Handle Repeated Search ---
    elif action_lower.startswith("search[") and action_lower.endswith("]"):
        query = agent_action[7:-1]
        logger.debug(f"Detected search action with query: '{query}'")
        if query == last_search_query:
            logger.debug(f"Detected repeated search query: '{query}'. Returning specific feedback without executing.")
            obs = "Repeating the exact same search query will return the same results. To find different items, please try refining your search query or explore the current results using 'click' actions."
            reward = 0.0
            done = False
            return obs, reward, done
        else:
            logger.debug(f"New search query: '{query}'. Executing search.")
            obs, reward, done = env.step(agent_action)
            # Update last_search_query only if the search was presumably successful
            if "invalid action" not in obs.lower(): # Basic check for success
                 logger.debug(f"Search successful. Updating last_search_query to: '{query}'")
                 last_search_query = query
            else:
                 logger.debug(f"Search action resulted in invalid action. Not updating last_search_query.")
            return obs, reward, done

    # --- Analysis Result 5 Refinement: Handle Purchase/Checkout Actions ---
    # Let purchase actions execute, then check the result ('done' flag).
    elif action_lower in ["click[buy now]", "click[add to cart]", "click[checkout]", "click[proceed to checkout]"]:
        logger.debug(f"Detected purchase/checkout action: {agent_action}. Executing env.step.")
        obs, reward, done = env.step(agent_action)
        logger.debug(f"Result from env.step for '{agent_action}': reward={reward}, done={done}, obs snippet='{obs[:150]}...'")

        # Check if the purchase action failed. Primary indicator is 'done' being False.
        # A successful purchase should end the episode (done=True).
        if not done:
            logger.warning(f"Purchase action '{agent_action}' failed because 'done' is False. Providing feedback about potentially missing options.")
            # Provide specific feedback guiding the agent to check for *all* required options.
            # Prepend the guidance to the original observation from the failed step.
            feedback = (f"Your action '{agent_action}' did not complete the purchase (the task is not 'done'). "
                        f"This often happens because not ALL required item options (e.g., size, color, style) were selected beforehand. "
                        f"Please carefully review the item page observation below for ANY options you haven't selected yet. "
                        f"Ensure ALL required options are chosen by using 'click[Option Name]' for each, then try '{agent_action}' again.\n\n"
                        f"Current Observation:\n{obs}")
            # Return the enhanced observation. Keep reward/done from the failed env.step (reward likely 0, done is False).
            return feedback, reward, done
        else:
            # Purchase action seems successful (done=True)
            logger.debug(f"Purchase action '{agent_action}' appears successful (done={done}).")
            return obs, reward, done

    # --- Default Action Processing ---
    else:
        # For any other action, execute it normally
        logger.debug(f"Executing default action: {agent_action}")
        obs, reward, done = env.step(agent_action)

        # Reset last_search_query if a non-search click action occurs,
        # particularly one that likely navigates to an item page.
        # Heuristic: Check for click actions that aren't known navigation/option clicks
        # and potentially look for product ID patterns if applicable.
        if action_lower.startswith("click["):
            # More specific check: Reset if it's likely a product title click
            # (e.g., not pagination, back, options, buy, etc.)
            # This is still heuristic without knowing exact state transitions.
            is_nav_or_option = any(kw in action_lower for kw in [
                "back", "prev", "next", "page", # Pagination/Nav
                "buy", "add", "cart", "checkout", # Purchase related
                "desc", "overview", "feature", "review", # Item details tabs
                "option", # Explicit option selection (though specific options might not contain 'option')
                # Add common option names if known, e.g., "size", "color", "style" ? Risky.
            ])
            # A more robust check might involve regex for typical product titles/IDs if pattern is known
            is_likely_product_click = not is_nav_or_option # Basic assumption

            if is_likely_product_click and last_search_query is not None:
                 logger.debug(f"Detected non-search/nav/option click action: '{agent_action}'. Resetting last_search_query.")
                 last_search_query = None

        return obs, reward, done
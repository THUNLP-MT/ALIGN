import logging
import re # Import re for potential future use

# Assume logger is configured elsewhere and passed to the function
# Example: logger = logging.getLogger(__name__)
# Assume env is an instance of the webshopEnv class with a step method
# Assume init_obs and task are strings provided as input

def InferRules(init_obs, task):
    """
    Contains the rules for environment and task execute logic.
    """
    # Per AnalysisAgent findings (Result 1, 2, 3, 4, 5, & 6) and instructions,
    # the identified misalignments relate to feedback during execution (WrapStep),
    # not the initial rule description provided to the agent.
    # Therefore, no changes are needed here based on the provided analysis.
    return "There is no rule for this environment."

def WrapStep(env, init_obs, task, agent_action: str, logger):
    """
    Process the agent action, execute it in the environment, and potentially refine
    the feedback based on identified misalignments.

    Refined based on:
    - AnalysisAgent Result 1 (Task ID 57): Provide specific feedback on partial success
      for purchase actions ('click[Buy]' or 'click[Buy Now]').
    - AnalysisAgent Result 2 (Task ID 80): Provide specific feedback when a 'search'
      action is attempted in an invalid state.
    - AnalysisAgent Result 3 (Task ID 107): Provide specific feedback when a 'click'
      action for an option (size, color, etc.) uses an incorrect format or targets
      an invalid item.
    - AnalysisAgent Result 4 (Task ID 182): Provide specific feedback when a 'click'
      action targets non-actionable bracketed items like [Features] or [Reviews].
    - AnalysisAgent Result 5 (Task ID 70): Provide specific feedback when 'click[Buy Now]'
      is attempted from an invalid state (e.g., Description page).
    - AnalysisAgent Result 6 (Task ID 178): Provide specific feedback when 'click[Buy]'
      is attempted after selecting an option, requiring a return to the main page first.
      (Integrated into Result 5 handler).
    """
    logger.debug(f"Processing agent action: {agent_action}")

    # Execute the action in the environment using the standard step function
    # This call changes the environment state.
    obs, reward, done = env.step(agent_action)
    logger.debug(f"Initial env.step result - obs length: {len(obs)}, reward: {reward}, done: {done}")

    # Normalize the agent action for case-insensitive comparison
    normalized_action = agent_action.strip().lower()
    # Define known non-option/non-product click arguments for easier checking
    # These are standard navigation/action commands.
    known_standard_clicks = [
        "click[back to search]",
        "click[prev page]",
        "click[next page]",
        "click[buy]",
        "click[buy now]",
        "click[desc/overview]",
        "click[back]" # Added 'back' as it's mentioned in feedback
    ]
    purchase_actions = ["click[buy]", "click[buy now]"]
    # Define specific non-actionable clicks identified in Analysis Result 4
    non_actionable_clicks = ["click[features]", "click[reviews]"]

    # --- Refinement based on AnalysisAgent Result 4 (Invalid Click on Features/Reviews) ---
    # Check specifically for attempts to click non-actionable items like [Features] or [Reviews]
    # that result in an invalid action. This check comes BEFORE the general invalid click handler (Result 3).
    if normalized_action in non_actionable_clicks and "Invalid action!" in obs and reward == 0.0 and not done:
        logger.debug(f"Detected invalid click action '{agent_action}' on non-actionable item ({normalized_action}). Replacing generic/misleading feedback based on AnalysisAgent Result 4.")
        # Provide specific feedback clarifying that these items are not clickable.
        obs = "Invalid action: 'Features' and 'Reviews' are not clickable actions. Only options explicitly listed as selectable (such as style, color, or size), [Description], or 'Buy Now' can be clicked. Please refer to the available options in the product detail page."
        logger.debug(f"Modified observation with specific invalid Features/Reviews click feedback. New obs length: {len(obs)}")
        # Return the refined observation and original reward/done status for the invalid action
        return obs, reward, done

    # --- Refinement based on AnalysisAgent Result 2 (Invalid Search Action) ---
    # Check if the action was a search attempt AND it resulted in the generic "Invalid action!" feedback.
    elif normalized_action.startswith("search[") and "Invalid action!" in obs and reward == 0.0 and not done:
        logger.debug(f"Detected invalid search action '{agent_action}' in the current state. Replacing generic feedback based on AnalysisAgent Result 2.")
        # Replace the generic observation with specific, actionable feedback
        obs = "Invalid action: You can only perform a search when you are on the Search page. Please click 'Back to Search' before searching again."
        logger.debug(f"Modified observation with specific invalid search feedback. New obs length: {len(obs)}")
        # Return the refined observation and original reward/done status for the invalid action
        return obs, reward, done

    # --- Refinement based on AnalysisAgent Results 5 & 6 (Invalid 'Buy Now'/'Buy' State) ---
    # Check if a purchase action was attempted, resulted in "Invalid action!", and the episode didn't end.
    # This suggests the action was invalid in the *current state* (e.g., trying to buy from Description page OR after selecting an option).
    elif normalized_action in purchase_actions and "Invalid action!" in obs and reward == 0.0 and not done:
        logger.debug(f"Detected invalid purchase action '{agent_action}' in the current state (likely not on main product page or after option selection). Replacing generic feedback based on AnalysisAgent Results 5 & 6.")
        # Provide refined feedback addressing both scenarios.
        obs = "Invalid action: You can only purchase a product from the main product detail page. If you just selected an option (like size or color), you might need to click [Back] to return to the main product page before clicking [Buy Now]. If you were viewing [Description] or other details, clicking [Overview] or [Back] might return you to the main page. Please check the available actions in the current observation to see how to return."
        logger.debug(f"Modified observation with specific invalid purchase state feedback (Results 5 & 6). New obs length: {len(obs)}")
        # Return the refined observation and original reward/done status for the invalid action
        return obs, reward, done

    # --- Refinement based on AnalysisAgent Result 3 (Invalid Option/Product Click Format) ---
    # Check if the action was a click attempt, resulted in "Invalid action!",
    # wasn't one of the standard navigation/action clicks, and wasn't one of the specific non-actionable clicks handled above.
    # This logic assumes that if a click[Something] fails with "Invalid action!", and "Something"
    # isn't a standard command or a known non-actionable item, it's likely an attempt to click a product or an option incorrectly.
    # Also ensure it wasn't an invalid purchase action state, which is handled above.
    elif (normalized_action.startswith("click[") and
            normalized_action not in known_standard_clicks and
            normalized_action not in non_actionable_clicks and # Ensure we don't override Result 4 feedback
            # No need to explicitly check against purchase_actions here, as the invalid state case (Results 5 & 6) is handled above.
            # If it's a purchase action and gets here, it means it didn't trigger the "Invalid action!" + reward=0 + done=False condition.
            "Invalid action!" in obs and
            reward == 0.0 and not done):
         logger.debug(f"Detected invalid click action '{agent_action}' that might be an incorrectly formatted option selection or invalid product click. Replacing generic feedback based on AnalysisAgent Result 3.")
         # Provide specific feedback guiding the agent on how to click options correctly,
         # or suggesting checking the product title format.
         obs = "Invalid action: Could not click the specified item. If you were trying to select an option (like size or color), ensure you use the exact text shown within the brackets, e.g., click[11-11.5 women] or click[p-blue]. If you were trying to click a product title, ensure it exactly matches the title shown."
         logger.debug(f"Modified observation with specific invalid click/option feedback. New obs length: {len(obs)}")
         # Return the refined observation and original reward/done status for the invalid action
         return obs, reward, done

    # --- Refinement based on AnalysisAgent Result 1 (Partial Success Feedback) ---
    # Check if the action is a recognized purchase action and resulted in partial success
    elif normalized_action in purchase_actions and done and 0 < reward < 1.0:
        logger.debug(f"Detected partial success (reward={reward}) for purchase action '{agent_action}'. Adding specific feedback based on AnalysisAgent Result 1.")
        # Construct the feedback message as suggested by AnalysisAgent's recommendation
        feedback = "Feedback: Partial credit awarded. The selected product meets the explicit task requirements (e.g., type, price), but does not fully satisfy all underlying criteria for maximum score. Consider if other product attributes (like size, brand, specific features, or ratings) might be relevant for optimal selection."
        # Prepend the feedback to the original observation
        obs = f"{feedback}\n\n{obs}"
        logger.debug(f"Modified observation with partial success feedback. New obs length: {len(obs)}")
    # Log other outcomes for recognized purchase actions when the episode ends
    elif normalized_action in purchase_actions and done:
        if reward == 1.0:
            logger.debug(f"Detected full success (reward={reward}) for purchase action '{agent_action}'.")
        elif reward == 0.0:
            # This case might occur if the purchase action itself was valid in the state,
            # but the purchased item was completely wrong according to task criteria,
            # leading to done=True and reward=0.
            # The invalid *state* case (Results 5 & 6) where done=False is handled above.
            logger.debug(f"Detected failure (reward={reward}) for purchase action '{agent_action}' when done={done}. The purchase was likely completed but the item was incorrect.")
        else:
            # Should not happen if reward is always 0, (0,1), or 1 when done=True, but log just in case.
             logger.warning(f"Detected unusual reward ({reward}) for purchase action '{agent_action}' when done={done}.")

    # Return the observation (potentially modified with feedback), reward, and done status
    return obs, reward, done
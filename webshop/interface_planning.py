import re
import logging
from typing import Tuple, Any, Dict
import io # Added for logger simulation if needed

# Assume logger is passed and configured externally
# Example configuration (if needed for testing):
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# log_stream = io.StringIO() # Capture logs
# if not logger.hasHandlers():
#     # handler = logging.StreamHandler() # Output to console
#     handler = logging.StreamHandler(log_stream) # Output to stream
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)

# --- Helper functions (unchanged from provided template) ---

def parse_task_constraints(task: str, logger: logging.Logger) -> Dict[str, Any]:
    """Parses price, quantity, and attribute constraints from the task string."""
    constraints: Dict[str, Any] = {}
    task_lower = task.lower()

    # Price (e.g., "under $30", "less than 30 dollars", "for $25")
    price_match_under = re.search(r'(?:under|less than|max|maximum)\s*\$?(\d+(?:\.\d+)?)', task_lower)
    price_match_limit = re.search(r'for\s*\$?(\d+(?:\.\d+)?)\s*(?:or less)?', task_lower)

    if price_match_under:
        constraints['price_max'] = float(price_match_under.group(1))
    elif price_match_limit:
         constraints['price_max'] = float(price_match_limit.group(1))

    # Quantity (e.g., "36 packets", "a pack of 12", "quantity 50")
    quantity_match = re.search(r'(\d+)\s*(?:count|pack|packets|items|quantity|pairs|sets)', task_lower)
    if quantity_match:
        constraints['quantity'] = int(quantity_match.group(1))

    # Attributes (keywords like "organic", "red", "cotton", "usda certified")
    attributes = []
    # Example specific attributes from Analysis 1
    if "usda certified organic" in task_lower:
        attributes.append("usda certified organic")
    # More specific attribute extraction based on common patterns
    if "black tea bags" in task_lower:
         attributes.append("black tea bags") # Example from analysis

    # Generic attributes (can be expanded)
    colors = re.findall(r'\b(red|blue|green|black|white|silver|grey|yellow|purple|orange|brown)\b', task_lower)
    attributes.extend(colors)
    materials = re.findall(r'\b(cotton|wool|silk|polyester|leather|metal|wood|plastic)\b', task_lower)
    attributes.extend(materials)

    # Add other common attributes if needed
    if "organic" in task_lower and "usda certified organic" not in attributes:
        attributes.append("organic")

    constraints['attributes'] = sorted(list(set(attributes)))
    if not constraints['attributes']: # Avoid empty list if no attributes found
        del constraints['attributes']

    logger.debug(f"Parsed constraints from task '{task}': {constraints}")
    return constraints

def parse_item_details(obs: str, logger: logging.Logger) -> Dict[str, Any]:
    """Parses price, quantity, and text content from the observation string."""
    details: Dict[str, Any] = {}
    obs_lower = obs.lower()

    # Price
    price_match = re.search(r'price:\s*\$(\d+(?:\.\d+)?)', obs_lower)
    if price_match:
        details['price'] = float(price_match.group(1))
    else:
        price_match_alt = re.search(r'\$(\d+(?:\.\d+)?)', obs_lower)
        if price_match_alt:
             details['price'] = float(price_match_alt.group(1))

    # Quantity
    quantity_match = re.search(r'(\d+)\s*(?:count|pack|items|pk)', obs_lower)
    if quantity_match:
        details['quantity'] = int(quantity_match.group(1))
    else:
        # Check title or options for quantity patterns
        title_match = re.search(r'page title:.*\(?pack of (\d+)\)?', obs_lower)
        if title_match:
             details['quantity'] = int(title_match.group(1))
        else:
            # Check for quantity in options list if available
            options_match = re.search(r'options:.*?(\d+)\s*(?:count|pack|items|pk)', obs_lower, re.DOTALL)
            if options_match:
                details['quantity'] = int(options_match.group(1))


    # Store the full lowercased text for attribute checking
    details['text'] = obs_lower

    logger.debug(f"Parsed details from observation: {details}")
    return details

# --- Refined Functions ---

def InferRules(init_obs: str, task: str) -> str:
    """
    Contains the rules for environment and task execute logic.
    Provides clearer guidance on the 'Buy Now' action state and constraints,
    notes potential state-dependent action restrictions (including post-failed-purchase states),
    and clarifies the behavior of 'Back to Search', especially from subpages.
    (Refined based on Analysis 1-10)
    """
    return """1. Only one action per turn in the format ACTION[Argument].
2. Available actions: search[Query], click[Back to Search], click[Prev/Next Page], click[Product Title], click[Option], click[Desc/Overview], click[Buy Now].
3. `click[Buy Now]` attempts to purchase the currently viewed item.
4. **State Requirement for Purchase:** `click[Buy Now]` can typically only be used when viewing the **main details of a specific item (Item page state)**. It is **not available** and will fail if used from other views like search results, or item sub-pages (e.g., Description, Features, Reviews). If attempted from an invalid state, you will receive a specific warning explaining the issue and guiding you back to the main item page. (See Rule 7 regarding 'Back to Search' after such a warning).
5. **Constraint Requirement for Purchase:** The purchase (`click[Buy Now]`) will only succeed if the item meets ALL requirements specified in the task (e.g., price limit, quantity, specific features like 'organic').
6. If `click[Buy Now]` is attempted on an item that does not meet all task requirements, the purchase will fail. You will receive a specific warning message indicating exactly which requirements were not met, and you will remain on the item page to choose a different item or adjust options. (See Rule 9 regarding action restrictions after this type of failure).
7. **`click[Back to Search]` Behavior:** This action usually returns you to the previous page (e.g., from an item page back to search results). **However, be aware:** Using `click[Back to Search]` from certain item sub-pages (like Description, Features, Reviews) or immediately after certain errors (like attempting 'Buy Now' from a sub-page) might **reset your session back to the initial search/instruction page**, requiring you to restart your search and navigation for that item. Pay close attention to feedback messages for guidance on how to navigate correctly in such cases. Also note the temporary restriction described in Rule 9.
8. **Other Action Restrictions:** Be aware that certain actions might become temporarily unavailable depending on the current state (e.g., after certain errors or actions). If an action is invalid in the current context, you will receive specific feedback guiding you on how to proceed.
9. **Post-Purchase Attempt Restrictions (Constraint Failure):** After attempting `click[Buy Now]`, if the purchase fails due to unmet constraints (Rule 6), the actions `search[...]` and `click[Back to Search]` will become temporarily unavailable. **This restriction is temporary.** To lift the restriction and enable `search[...]` or `click[Back to Search]` again, you must first perform one different valid action (any action *other than* `search[...]` or `click[Back to Search]`, e.g., 'click[Prev/Next Page]', 'click[Option]', 'click[Desc/Overview]'). **Successfully performing such an action will lift the restriction.** You will receive a specific warning if you attempt a restricted action while the restriction is active, explaining how to proceed."""


def WrapStep(env: Any, init_obs: str, task: str, agent_action: str, logger: logging.Logger) -> Tuple[str, float, bool]:
    """
    Process the agent action, handling invalid actions and constraint checks based on analysis results.
    Provides specific feedback for invalid actions (including state errors for Buy Now and resets for Back to Search)
    and unmet constraints during Buy Now attempts.
    Incorporates explicit state tracking for post-failed-buy restrictions (Analysis 3, 4, 6, 7)
    by using a flag `_post_failed_buy_restriction_active` on the `env` object.
    Ensures the restriction is correctly lifted after an intermediate valid action (Analysis 7).
    Checks for session resets caused by 'Back to Search' from subpages (Analysis 8, 10).
    Assumes modification of the `env` object is permissible.

    Args:
        env: The environment instance. (Expected to allow adding attributes).
        init_obs: The initial observation string (used to detect session resets).
        task: The task description string.
        agent_action: The action chosen by the agent.
        logger: Logger instance for debugging.

    Returns:
        A tuple containing the observation (str), reward (float), and done (bool).
    """
    logger.debug(f"Processing action: {agent_action}")
    logger.debug(f"Initial Observation provided: {init_obs[:100]}...") # Log start of init_obs for comparison

    # Initialize state flag if it doesn't exist
    if not hasattr(env, '_post_failed_buy_restriction_active'):
        env._post_failed_buy_restriction_active = False
        logger.debug("Initialized env._post_failed_buy_restriction_active to False.")
    else:
        # Log current state of the flag at the beginning of processing
        logger.debug(f"Start of processing. Current restriction flag state: {env._post_failed_buy_restriction_active}")


    normalized_action = agent_action.strip().lower()
    is_search_action = normalized_action.startswith("search[")
    is_back_to_search_action = normalized_action == "click[back to search]"
    is_buy_action = normalized_action.startswith("click[buy") # Covers "buy now" etc.

    # --- State Tracking Logic for Post-Failed-Buy Restriction ---
    # Check if restriction is active and action is restricted
    if env._post_failed_buy_restriction_active and (is_search_action or is_back_to_search_action):
        action_type = "Search" if is_search_action else "Back to Search"
        feedback = (f"Warning: '{action_type}' action failed. This action is temporarily restricted because a previous purchase attempt failed "
                    f"due to unmet requirements (constraint failure). To enable '{action_type}' again, please first perform one navigation action "
                    f"like 'click[Prev/Next Page]' or interact with the current item (e.g., 'click[Option]', 'click[Desc/Overview]'). See Rule 9.")
        logger.info(f"Blocked restricted action '{agent_action}' due to active flag. Providing feedback (Analysis 3, 4, 6). Flag remains True.")
        # Return feedback indicating state hasn't changed. Assume current obs is needed for context.
        # We need the *current* observation, not init_obs. Since we haven't called env.step, we don't have a new one.
        # This scenario implies the agent is trying an action *before* getting new obs, which shouldn't happen in the loop.
        # Let's assume we need to return the *previous* observation. This requires storing it, which complicates things.
        # A simpler approach for the benchmark: return the feedback prepended to a generic message about state.
        modified_obs = feedback + "\n\n" + "Current page state remains unchanged. Please choose a different action based on the previous observation and this warning."
        # Alternatively, if env has a way to get current obs without stepping: obs = env.get_observation()
        # If not, this feedback is the best we can do without modifying the main loop structure.
        return modified_obs, 0.0, False # Return 0 reward, not done

    # If restriction is active but action is allowed, clear the restriction flag *before* stepping
    # This addresses Analysis 7 by ensuring the flag is cleared upon performing the required intermediate action.
    if env._post_failed_buy_restriction_active and not (is_search_action or is_back_to_search_action):
        logger.debug(f"Restriction flag was active. Action '{agent_action}' is allowed. Clearing restriction flag *before* executing action (Analysis 7 fix).")
        env._post_failed_buy_restriction_active = False
    # --- End State Tracking Logic ---

    # 1. Execute the action in the environment
    # This is now only called if the action wasn't blocked by the restriction flag.
    logger.debug(f"Executing env.step({agent_action}). Restriction flag is currently: {env._post_failed_buy_restriction_active}")
    original_obs, original_reward, original_done = env.step(agent_action)
    logger.debug(f"Environment response: done={original_done}, reward={original_reward}")
    # Log flag state *after* env.step but before any further modification
    logger.debug(f"After env.step. Restriction flag is currently: {env._post_failed_buy_restriction_active}")
    logger.debug(f"Observation received: {original_obs[:200]}...") # Log start of received obs

    # Initialize final results with original ones
    final_obs = original_obs
    final_reward = original_reward
    final_done = original_done

    # 2. Check for "Invalid action" feedback from the environment (for reasons *other* than the restriction flag)
    # Note: If the action was blocked by the flag, we returned earlier. This handles invalid actions reported by env.step itself.
    if re.search(r'invalid action', original_obs, re.IGNORECASE):
        logger.info(f"Detected 'Invalid action' response from env.step for action: {agent_action}")
        feedback = ""

        if is_buy_action:
            # Analysis Result 2 & 9: Invalid state for Buy Now (e.g., buying from Features/Description page)
            feedback = ("Warning: 'Buy Now' action failed. This action is only available from the **main item detail page**, "
                        "not from sub-pages (like Description, Features, Reviews) or search results. Please navigate back to the item's main page "
                        "(e.g., using 'click[Desc/Overview]' if you are on a sub-page, or clicking the item title from search results) if you wish to purchase it. "
                        "Note: Using 'click[Back to Search]' immediately after this warning might reset you to the initial search page, "
                        "requiring you to find the item again (See Rule 7). Consider navigating back to the main item page first.")
            logger.info("Providing feedback for 'Buy Now' invalid state (Analysis 2 & 9), including 'Back to Search' warning (Analysis 5, 8, 10).")
        # Note: Cases for search/back-to-search being invalid due to restriction are handled *before* env.step now.
        # This block handles other potential reasons for these actions being invalid, if any.
        elif is_search_action:
             feedback = (f"Warning: The action '{agent_action}' is invalid in the current context (reason other than post-failed-buy restriction). "
                         f"Please check the available actions and the current page state.")
             logger.warning(f"Unhandled 'Invalid action' for SEARCH action: {agent_action}. Providing generic feedback.")
        elif is_back_to_search_action:
             # This could be an invalid state for Back to Search other than the restriction or the subpage reset case.
             feedback = (f"Warning: The action '{agent_action}' is invalid in the current context (reason other than post-failed-buy restriction or subpage reset). "
                         f"Please check the available actions and the current page state.")
             logger.warning(f"Unhandled 'Invalid action' for BACK TO SEARCH action: {agent_action}. Providing generic feedback.")
        else:
            # Generic invalid action feedback for other actions
            feedback = f"Warning: The action '{agent_action}' is invalid in the current context. Please check the available actions and the current page state. Refer to the environment rules if needed."
            logger.warning(f"Unhandled 'Invalid action' for action: {agent_action}. Providing generic feedback.")

        # Prepend feedback and override results
        final_obs = feedback + "\n\n" + original_obs
        final_reward = 0.0
        final_done = False
        # Log flag state before returning after invalid action
        logger.debug(f"Returning after 'Invalid action' from env.step. Restriction flag state: {env._post_failed_buy_restriction_active}")
        return final_obs, final_reward, final_done

    # 3. Check for session reset after 'click[Back to Search]' (Analysis 8, 10)
    # This check happens *after* the invalid action check, as a reset isn't necessarily an "invalid action" from the env's perspective.
    if is_back_to_search_action:
        # Check if the observation looks like the initial instruction page
        # Using the presence of "Instruction:" as a marker for the initial state.
        if "Instruction:" in original_obs and original_obs.strip().startswith(init_obs.strip()[:50]): # Check start similarity too
            feedback = ("Warning: Using 'click[Back to Search]' from the previous page has reset your session to the initial instruction page. "
                        "This can happen when using 'Back to Search' from item sub-pages (like Description, Features, Reviews) or after certain errors (See Rule 7). "
                        "To proceed with the task, you may need to start your search and navigation again. To purchase an item, ensure you navigate to its main detail page before using 'click[Buy Now]'.")
            logger.info("Detected session reset after 'click[Back to Search]'. Prepending feedback (Analysis 8, 10).")
            final_obs = feedback + "\n\n" + original_obs
            # Reset doesn't necessarily mean failure, so keep original reward/done unless they indicate failure.
            # However, a reset usually implies the task isn't done. Let's force done=False.
            final_done = False
            # Log flag state before returning after reset detection
            logger.debug(f"Returning after 'Back to Search' reset detection. Restriction flag state: {env._post_failed_buy_restriction_active}")
            return final_obs, final_reward, final_done # Return potentially modified obs, original reward, forced done=False

    # 4. If the action was 'click[Buy Now]' and *not* an invalid state error, check constraints
    if is_buy_action:
        logger.debug("Action 'Buy Now' was accepted by env state. Now checking task constraints.")

        constraints = parse_task_constraints(task, logger)
        item_details = parse_item_details(original_obs, logger)
        unmet_constraints = []

        # Check if the purchase was actually successful according to the environment *before* checking constraints
        # If original_done is True and reward > 0, assume success regardless of our parsing.
        if original_done and original_reward > 0:
             logger.info("Environment indicates successful purchase (done=True, reward>0). Skipping constraint check override.")
             # Ensure flag is false on success
             if env._post_failed_buy_restriction_active:
                 logger.debug("Clearing restriction flag as Buy Now was successful according to env.")
                 env._post_failed_buy_restriction_active = False
             return original_obs, original_reward, original_done

        # If not clearly successful, proceed with constraint checks
        if not item_details and not original_done:
             logger.warning("Could not parse item details from observation after 'Buy Now' attempt, and episode is not done. Cannot verify constraints.")
             # If we cannot parse details, we cannot confirm constraints are met. Treat as failure if constraints exist.
             if constraints:
                 unmet_constraints.append("Could not parse item details from the current view to verify constraints")
        elif item_details:
            # Perform constraint checks
            if 'price_max' in constraints:
                if 'price' in item_details:
                    if item_details['price'] > constraints['price_max']:
                        unmet_constraints.append(f"Price is too high (item price ${item_details['price']:.2f} exceeds budget limit of ${constraints['price_max']:.2f})")
                else: unmet_constraints.append(f"Could not verify price (budget limit is ${constraints['price_max']:.2f}, but item price was not found)")
            if 'quantity' in constraints:
                if 'quantity' in item_details:
                    if item_details['quantity'] != constraints['quantity']:
                        unmet_constraints.append(f"Incorrect quantity (task requires {constraints['quantity']}, item has {item_details['quantity']})")
                else: unmet_constraints.append(f"Could not verify quantity (task requires {constraints['quantity']}, but item quantity was not found)")
            if 'attributes' in constraints:
                if 'text' in item_details:
                    item_text_lower = item_details['text']
                    missing_attributes = []
                    for attr in constraints['attributes']:
                        # Use word boundary regex for more robust attribute matching
                        if not re.search(r'\b' + re.escape(attr.lower()) + r'\b', item_text_lower):
                             missing_attributes.append(f"'{attr}'")
                    if missing_attributes: unmet_constraints.append(f"Missing required attributes: {', '.join(missing_attributes)}")
                else: unmet_constraints.append(f"Could not verify attributes ({', '.join(constraints['attributes'])}) because item description/details were not found")

        # Check if any constraints were unmet
        if unmet_constraints:
            # Analysis Result 1: Constraints unmet for Buy Now
            feedback = (f"Warning: 'Buy Now' action aborted. The item does not meet all task requirements. Unmet constraints: {'; '.join(unmet_constraints)}. "
                        f"You remain on the item page. Please find an item that satisfies all conditions or navigate elsewhere. "
                        f"Note: Actions 'Search' and 'Back to Search' are now temporarily restricted. Perform one other valid action (like 'click[Prev/Next Page]', 'click[Option]', 'click[Desc/Overview]') to enable them again (see Rule 9).")
            logger.info(f"Constraint check failed for 'Buy Now'. Setting restriction flag. Overriding results (Analysis 1). Feedback: {feedback}")

            # *** SET RESTRICTION FLAG ***
            env._post_failed_buy_restriction_active = True
            logger.debug("Set env._post_failed_buy_restriction_active to True due to Buy Now constraint failure.")

            final_obs = feedback + "\n\n" + original_obs
            final_reward = 0.0
            final_done = False # Override done status
            return final_obs, final_reward, final_done
        else:
            # Constraints met or no constraints to check, but env didn't signal success. Use original results.
            logger.info("Constraint check passed (or not applicable) for 'Buy Now', but env did not signal success. Using original env.step results.")
            # Ensure flag is false if purchase constraints were met/NA (it might have been set previously)
            if env._post_failed_buy_restriction_active:
                 logger.debug("Clearing restriction flag as Buy Now constraint check passed or was not applicable (though env didn't signal success).")
                 env._post_failed_buy_restriction_active = False
            # Log flag state before returning after Buy Now (constraints met/NA, but no env success)
            logger.debug(f"Returning after Buy Now (constraints met/NA, no env success). Restriction flag state: {env._post_failed_buy_restriction_active}")
            return original_obs, original_reward, original_done # Return original results

    # 5. For all other actions that were not 'Invalid action', not 'Back to Search' reset, and not 'Buy Now'
    logger.debug(f"Action '{agent_action}' executed successfully (not Buy Now, not Invalid Action, not Back to Search reset). Returning original results.")
    # The flag was already cleared before env.step if it needed to be (Analysis 7 fix).
    # Log final flag state before returning
    logger.debug(f"Returning after successful non-Buy/non-reset/non-invalid action. Restriction flag state: {env._post_failed_buy_restriction_active}")
    return final_obs, final_reward, final_done # Return the (potentially unmodified) results
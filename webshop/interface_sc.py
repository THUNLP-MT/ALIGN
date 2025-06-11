import re
import logging

# Assume logger is configured elsewhere and passed to the function.
# Example: logger = logging.getLogger(__name__)

def InferRules(init_obs, task):
    """
    Contains the rules for environment and task execute logic.
    Reflects the requirement to select options before buying.

    Args:
        init_obs: str, the initial observation from the environment.
        task: str, the task description.

    Returns:
        str: A string describing the environment rules.
    """
    # Base rules
    base_rule = """1. Only use actions listed in the Action Space.
2. Provide one action per turn in the format ACTION[Argument].
3. Wait for environment feedback before providing the next action."""

    # Rule added based on Analysis Result 1
    buy_rule = """4. On an item page, if options like color or size are presented, you must select them using 'click[Option]' before you can use 'click[Buy Now]'."""

    return f"{base_rule}\n{buy_rule}"

def WrapStep(env, init_obs, task, agent_action: str, logger):
    """
    Process the agent action according to the refined strategy:
    - Executes the action using env.step first.
    - If the action was 'click[Buy Now]', analyzes the resulting observation ('obs').
    - Uses expanded failure indicators and checks for persistent option elements
      to detect failure due to missing options.
    - If 'obs' indicates failure due to missing options (and the action didn't succeed anyway),
      appends specific feedback to 'obs' and returns modified_obs, 0.0, False.
    - Otherwise, returns the original observation, reward, and done status from env.step.

    Args:
        env: The environment object, capable of executing env.step(action).
        init_obs: The initial observation (unused in this specific logic but part of signature).
        task: The task description (unused in this specific logic but part of signature).
        agent_action: The action string from the agent.
        logger: Logger object for logging debug and info messages.

    Returns:
        Tuple[str, float, bool]: The next observation, reward, and done status.
    """
    logger.debug(f"Executing agent action: {agent_action}")

    # Step 1: Execute the action in the environment unconditionally
    try:
        obs, reward, done = env.step(agent_action)
        logger.debug(f"Received from env.step - Obs snippet: {obs[:200]}..., Reward: {reward}, Done: {done}")
    except Exception as e:
        logger.error(f"Error during env.step({agent_action}): {e}")
        error_obs = f"An error occurred while executing action '{agent_action}': {e}"
        # Return error observation, 0 reward, and potentially end the episode
        return error_obs, 0.0, True # Or False, depending on desired behavior on env error

    # Step 2: Normalize the action for checking
    normalized_action = agent_action.strip().lower()

    # Step 3: Check if the action was 'click[buy now]'
    if normalized_action == "click[buy now]":
        logger.debug("Action was 'click[buy now]'. Analyzing observation for option requirement failure.")

        # Step 4a: Analyze the resulting observation for explicit failure indicators (Expanded List)
        failure_indicators = [
            "please select options", "select required options", "choose an option",
            "select color", "select size", "missing required fields",
            "must make a selection", "select product options", "you must select",
            "choose your size", "choose your color", "option is required",
            "select an option", "please choose", "required option missing"
            # Add more specific indicators based on observed environment behavior
        ]
        obs_lower = obs.lower() # Convert observation to lower case once for case-insensitive checks
        failed_due_to_options_keyword = False
        for indicator in failure_indicators:
            if indicator in obs_lower:
                failed_due_to_options_keyword = True
                logger.info(f"Detected potential failure keyword indicator '{indicator}' in observation after 'click[buy now]'.")
                break

        # Step 4b: Analyze observation for persistent option selection elements (More Robust Check)
        # Looks for HTML patterns suggesting options are still present and selectable
        # This assumes 'obs' contains some HTML structure. Adjust patterns as needed.
        option_element_patterns = [
            r'<select\s+.*?(name|id)=["\']?(option|size|color|variation|sku)["\']?.*?>', # Select dropdowns
            r'<input\s+.*?type=["\']?radio["\']?.*?(name|id)=["\']?(option|size|color|variation|sku)["\']?.*?>', # Radio buttons
            r'<input\s+.*?type=["\']?checkbox["\']?.*?(name|id)=["\']?(option|size|color|variation|sku)["\']?.*?>', # Checkboxes (less common for required single choices)
            r'button.*?(size|color|option)', # Buttons used for selection
            r'variant-selector', # Common class/id names
            r'product-options'
        ]
        options_still_present = False
        for pattern in option_element_patterns:
            # Use re.IGNORECASE for case-insensitivity in HTML attributes/tags
            if re.search(pattern, obs, re.IGNORECASE):
                options_still_present = True
                logger.info(f"Detected persistent option element pattern '{pattern}' in observation after 'click[buy now]'.")
                break

        # Step 4c: Check if the action might have succeeded despite indicators/elements
        success_indicators = ["shopping cart", "your cart", "checkout", "order summary", "item added to cart", "proceed to checkout"]
        action_succeeded = any(indicator in obs_lower for indicator in success_indicators)

        # Step 5: Provide feedback if failure detected (by keyword OR structure), action didn't succeed
        # Combine keyword and structural checks for better robustness
        failure_detected = failed_due_to_options_keyword or options_still_present

        if failure_detected and not action_succeeded:
            logger.info(f"Action '{agent_action}' appears to have failed due to missing options (keyword: {failed_due_to_options_keyword}, structure: {options_still_present}) and did not lead to success state. Modifying response.")
            feedback = "\n\nFeedback: Invalid action. Please select required options (e.g., color, size) before clicking 'Buy Now'."
            modified_obs = obs + feedback
            # Override reward and done status as per requirement to guide the agent
            return modified_obs, 0.0, False
        else:
             # Action was 'click[buy now]' but either succeeded, failed for other reasons,
             # or the failure wasn't detected by our refined heuristics. Return original results.
             if action_succeeded:
                 logger.debug("Action 'click[buy now]' seems to have succeeded based on success indicators. Returning original env.step results.")
             elif failure_detected: # Implies action_succeeded was false
                 logger.debug("Action 'click[buy now]' failed due to options, but action_succeeded was false (or env returned done=True). Returning original env.step results.")
             else: # failure_detected is False
                 logger.debug("Action 'click[buy now]' executed. No definitive option failure detected. Returning original env.step results.")
             return obs, reward, done

    else:
        # Step 6: Default Return for actions other than 'click[buy now]'
        # Action was not 'click[buy now]', return the original results directly.
        logger.debug(f"Action '{agent_action}' is not 'click[buy now]'. Returning original env.step results.")
        return obs, reward, done
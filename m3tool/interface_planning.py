import re
import logging
from typing import Tuple, Any, Dict # Assuming Task environment class is defined elsewhere

# Assume logger is properly initialized elsewhere in the system
# Example: logger = logging.getLogger("EnvironmentLogic")

# Define the specific task names for Analysis Results 2 and 3
TRADE_CALCULATOR_MOST_PROFITABLE_TASK_NAME = 'trade_calculator/most_profitable_conversion_rate' # Task ID 2-4
TRADE_CALCULATOR_BULK_DISCOUNT_TASK_NAME = 'trade_calculator/bulk_trade_quantity_discount' # Task ID 2-5 (Updated name)

# Mapping from task_type_idx to task type name (for clarity)
TASK_TYPE_MAP: Dict[int, str] = {
    0: 'message_decoder',
    1: 'cryptobotanists_plant_dna_sequencer',
    2: 'trade_calculator',
    3: 'travel_itinerary_planning',
    4: 'web_browsing',
}

def InferRules(task_name: str, task_type_idx: int) -> str:
    """
    Contains the rules for environment and task execute logic for different task types.
    Provides specific rules for tasks where misalignments have been identified.
    """
    task_name_lower = task_name.lower()

    # Task ID 2-4: trade_calculator/most_profitable_conversion_rate
    if task_type_idx == 2 and task_name_lower == TRADE_CALCULATOR_MOST_PROFITABLE_TASK_NAME.lower():
        # Rule derived from Analysis Result 2
        return (
            f"Rule: For this task ('{task_name}'), you are given a base price per unit (300) "
            "and need to calculate the final trade value for 1000 units considering multiple conversion rates (2.5, 2.3, 2.7) "
            "and a tariff (10%).\n"
            "IMPORTANT: You must first calculate the total base price for all 1000 units (base price per unit * 1000). "
            "Then, convert this *total* amount using the most profitable conversion rate. "
            "Finally, apply the 10% tariff to the *total* converted value to determine the final answer. "
            "Do not perform currency conversion or tariff calculations on a per-unit basis after the initial step of calculating the total base price."
        )

    # Task ID 2-5: trade_calculator/bulk_trade_quantity_discount (Updated name check)
    elif task_type_idx == 2 and task_name_lower == TRADE_CALCULATOR_BULK_DISCOUNT_TASK_NAME.lower():
        # Rule derived from Analysis Result 3
        base_price = 1500
        num_units = 2500
        discount_threshold = 2000
        discount_rate = 0.05
        conversion_rate = 1.8
        return (
            f"Rule: For this task ('{task_name}'), you need to calculate the final trade value for {num_units} units "
            f"with a base price of {base_price} per unit and a conversion rate of {conversion_rate}.\n"
            f"IMPORTANT: A bulk discount of {discount_rate*100}% applies if {discount_threshold} units or more are purchased. "
            f"Since {num_units} >= {discount_threshold}, you must first apply the discount to the base price per unit. "
            f"Then, calculate the total price for all {num_units} units using the discounted price. "
            f"Finally, convert this *total* discounted amount using the conversion rate ({conversion_rate}). "
            "Do not convert the original base price per unit directly."
        )

    # Task Type 3: travel_itinerary_planning (Added based on Analysis Result 4 context)
    elif task_type_idx == 3:
        # General rule/reminder for travel planning tasks, highlighting find_flights
        return (
            "Rule: For travel planning tasks, ensure you provide all required arguments for the tools. "
            "For example, the 'find_flights' tool requires 'from_location', 'to_location', and 'date'. "
            "Use the format: Action: find_flights, <from_location>, <to_location>, <date> End Action."
        )

    # Task Type 4: web_browsing (Added based on Analysis Result 5 context)
    elif task_type_idx == 4:
        # General rule/reminder for web browsing tasks, highlighting click_url format
        return (
            "Rule: For web browsing tasks, ensure you use the correct format for tool invocation. "
            "Arguments should be separated from the tool name by a comma. "
            "For example, to click a URL like '/team', use 'Action: click_url, /team End Action'. "
            "Do not use formats like 'Action: click_url('/team') End Action' where the argument is enclosed in parentheses and quotes immediately after the tool name."
        )


    # Default rule for other tasks
    return "There is no specific rule for this environment. Please follow the instructions and tool descriptions carefully."

def WrapStep(env: Any, task_name: str, instruction: str, agent_action: str, logger: logging.Logger) -> Tuple[str, float, bool]:
    """
    Process the agent action:
    - Checks for specific actions violating the Environment World Model for identified tasks.
    - Provides specific feedback for such violations without calling env.step, guiding the agent.
    - Otherwise, calls env.step and returns the results.

    Args:
        env: The task environment instance.
        task_name: The name of the current task.
        instruction: The task instruction (unused in this specific refinement but available).
        agent_action: The action string from the agent.
        logger: Logger instance for debugging.

    Returns:
        A tuple containing:
        - obs (str): The observation feedback for the agent.
        - reward (float): The reward obtained (0.0 for intercepted invalid actions).
        - done (bool): Whether the task is finished (False for intercepted invalid actions).
    """
    logger.debug(f"Processing agent action for task '{task_name}': {agent_action}")
    task_name_lower = task_name.lower()
    # Determine task_type_idx from task_name or pass it if available
    # For this example, we'll infer it based on known names or assume it's passed correctly.
    # A robust implementation might involve a lookup based on task_name.
    task_type_idx = -1 # Placeholder
    for idx, type_name in TASK_TYPE_MAP.items():
        # Use 'in' for broader matching if task names might have prefixes/suffixes
        # Or use exact match if task names are precise identifiers within a type
        # Example using 'in': if type_name in task_name_lower:
        # Example using startswith for type prefix: if task_name_lower.startswith(type_name):
        # Using 'in' based on original code's apparent logic
        if type_name in task_name_lower:
             task_type_idx = idx
             break
    # If type not found by 'in', try matching based on task ID prefix if available (e.g., "4-5")
    if task_type_idx == -1:
        task_id_match = re.match(r"(\d+)-\d+", task_name)
        if task_id_match:
            try:
                inferred_idx = int(task_id_match.group(1))
                if inferred_idx in TASK_TYPE_MAP:
                    task_type_idx = inferred_idx
            except ValueError:
                pass # Ignore if the first part is not a valid integer index

    logger.debug(f"Inferred task_type_idx: {task_type_idx} for task_name: {task_name}")


    # --- Logic specific to Task ID 2-4 (trade_calculator/most_profitable_conversion_rate) ---
    if task_type_idx == 2 and task_name_lower == TRADE_CALCULATOR_MOST_PROFITABLE_TASK_NAME.lower():
        # Define known parameters for this specific task based on Analysis Result 2
        base_price_per_unit = 300.0
        num_units = 1000
        conversion_rates = [2.5, 2.3, 2.7]
        # Calculate possible per-unit converted values for checking agent's intermediate steps
        possible_per_unit_conversions = [base_price_per_unit * rate for rate in conversion_rates] # [750.0, 690.0, 810.0]

        # Parse the action string to identify tool calls
        action_match = re.match(r"Action:\s*(\w+)(?:,\s*(.*?))?\s*End Action", agent_action, re.IGNORECASE | re.DOTALL)

        if action_match:
            tool_name = action_match.group(1)
            args_str = action_match.group(2)
            args = [a.strip() for a in args_str.split(',')] if args_str else []
            logger.debug(f"Parsed tool: {tool_name}, args: {args}")

            try:
                # Check 1: Is agent trying to convert the per-unit price?
                if tool_name.lower() == 'convert_currency':
                    if len(args) >= 1:
                        amount = float(args[0])
                        # Use a small tolerance for float comparison
                        if abs(amount - base_price_per_unit) < 1e-6:
                            obs = (f"Feedback: Invalid action. You seem to be converting the price per unit ({amount}). "
                                   f"As per the rules, you must first calculate the total price for all {num_units} units "
                                   f"(i.e., {base_price_per_unit} * {num_units}) before converting currency.")
                            logger.debug(f"Intercepted '{tool_name}' on per-unit price for task '{task_name}'. Feedback: {obs}")
                            return obs, 0.0, False # Return feedback, no reward, task not done

                # Check 2: Is agent trying to calculate tariff on a per-unit converted value?
                elif tool_name.lower() == 'calculate_tariff':
                     if len(args) >= 1:
                        converted_value = float(args[0])
                        # Check if the input value matches any of the calculated per-unit conversions
                        is_per_unit_value = any(abs(converted_value - val) < 1e-6 for val in possible_per_unit_conversions)
                        if is_per_unit_value:
                            obs = (f"Feedback: Invalid action. You seem to be calculating the tariff on a per-unit converted value ({converted_value}). "
                                   f"As per the rules, you must apply the tariff only to the *total* converted value for all {num_units} units.")
                            logger.debug(f"Intercepted '{tool_name}' on per-unit value for task '{task_name}'. Feedback: {obs}")
                            return obs, 0.0, False # Return feedback, no reward, task not done

                # Check 3: Is agent trying to estimate final value based on a per-unit value?
                elif tool_name.lower() == 'estimate_final_value':
                     if len(args) >= 1:
                        value_before_tariff = float(args[0])
                        # Check if the input value matches any of the calculated per-unit conversions
                        is_per_unit_value = any(abs(value_before_tariff - val) < 1e-6 for val in possible_per_unit_conversions)
                        if is_per_unit_value:
                            obs = (f"Feedback: Invalid action. You seem to be estimating the final value based on a per-unit value ({value_before_tariff}). "
                                   f"As per the rules, the final value must be calculated based on the *total* value for all {num_units} units after conversion and tariff.")
                            logger.debug(f"Intercepted '{tool_name}' on per-unit value for task '{task_name}'. Feedback: {obs}")
                            return obs, 0.0, False # Return feedback, no reward, task not done

            except (ValueError, IndexError) as e:
                # Log error if arguments are not as expected (e.g., not numbers, wrong count)
                logger.warning(f"Error parsing arguments for tool '{tool_name}' in action '{agent_action}' for task '{task_name}': {e}. Allowing env.step to handle.")
                pass # Fall through to the default env.step call

        # If the action was not an intercepted tool call for this specific task, proceed with the default behavior.
        logger.debug(f"Action not intercepted for task '{task_name}'. Proceeding to env.step.")


    # --- Logic specific to Task ID 2-5 (trade_calculator/bulk_trade_quantity_discount) --- (Updated name check)
    elif task_type_idx == 2 and task_name_lower == TRADE_CALCULATOR_BULK_DISCOUNT_TASK_NAME.lower():
        # Define known parameters for this specific task based on Analysis Result 3
        BASE_PRICE = 1500.0
        NUM_UNITS = 2500
        DISCOUNT_THRESHOLD = 2000
        DISCOUNT_RATE = 0.05
        # CONVERSION_RATE = 1.8 # Not needed for check, but defined for context

        # Parse the action string to identify tool calls
        action_match = re.match(r"Action:\s*(\w+)(?:,\s*(.*?))?\s*End Action", agent_action, re.IGNORECASE | re.DOTALL)

        if action_match:
            tool_name = action_match.group(1)
            args_str = action_match.group(2)
            args = [a.strip() for a in args_str.split(',')] if args_str else []
            logger.debug(f"Parsed tool: {tool_name}, args: {args}")

            try:
                # Check: Is agent trying to convert the base per-unit price directly?
                if tool_name.lower() == 'convert_currency':
                    if len(args) >= 1:
                        amount_to_convert = float(args[0])
                        # Use a small tolerance for float comparison
                        if abs(amount_to_convert - BASE_PRICE) < 1e-6:
                            # Check if discount should apply (it should in this task)
                            if NUM_UNITS >= DISCOUNT_THRESHOLD:
                                discounted_price_per_unit = BASE_PRICE * (1 - DISCOUNT_RATE)
                                obs = (f"Feedback: Invalid action. You seem to be converting the base price per unit ({BASE_PRICE}). "
                                       f"For purchases of {DISCOUNT_THRESHOLD} units or more ({NUM_UNITS} units in this case), "
                                       f"you must first apply the {DISCOUNT_RATE*100:.0f}% discount to the base price per unit "
                                       f"(New price: {BASE_PRICE} * (1 - {DISCOUNT_RATE}) = {discounted_price_per_unit:.1f}). "
                                       f"Then, calculate the total price for all {NUM_UNITS} units before converting currency.")
                                logger.debug(f"Intercepted '{tool_name}' on base per-unit price for task '{task_name}'. Feedback: {obs}")
                                return obs, 0.0, False # Return feedback, no reward, task not done
                            else:
                                # This case shouldn't happen for task 2-5 based on its definition, but included for robustness
                                logger.debug(f"Agent converting base price, but discount condition ({NUM_UNITS} < {DISCOUNT_THRESHOLD}) not met. Allowing action for task '{task_name}'.")
                                # Fall through to env.step as converting base price is correct if no discount applies

            except (ValueError, IndexError) as e:
                 # Log error if arguments are not as expected (e.g., not numbers, wrong count)
                logger.warning(f"Error parsing arguments for tool '{tool_name}' in action '{agent_action}' for task '{task_name}': {e}. Allowing env.step to handle.")
                pass # Fall through to the default env.step call

        # If the action was not an intercepted tool call for this specific task, proceed with the default behavior.
        logger.debug(f"Action not intercepted for task '{task_name}'. Proceeding to env.step.")

    # --- Logic specific to Task Type 3 (travel_itinerary_planning) --- (Added based on Analysis Result 4)
    elif task_type_idx == 3:
        # Parse the action string to identify tool calls
        action_match = re.match(r"Action:\s*(\w+)(?:,\s*(.*?))?\s*End Action", agent_action, re.IGNORECASE | re.DOTALL)

        if action_match:
            tool_name = action_match.group(1)
            args_str = action_match.group(2)
            args = [a.strip() for a in args_str.split(',')] if args_str else []
            logger.debug(f"Parsed tool: {tool_name}, args: {args}")

            # Check: Is agent using find_flights with incorrect number of arguments?
            if tool_name.lower() == 'find_flights':
                # find_flights requires 3 arguments: from_location, to_location, date
                if len(args) != 3:
                    obs = ("Feedback: Invalid action. You must specify all three required arguments for 'find_flights': "
                           "'from_location', 'to_location', and 'date'. "
                           "Use the format: Action: find_flights, <from_location>, <to_location>, <date> End Action. "
                           "For example: Action: find_flights, E, D, 2023-08-15 End Action.")
                    logger.debug(f"Intercepted '{tool_name}' with incorrect number of arguments ({len(args)}) for task type 3. Feedback: {obs}")
                    return obs, 0.0, False # Return feedback, no reward, task not done
                else:
                    logger.debug(f"Correct number of arguments ({len(args)}) provided for '{tool_name}'. Proceeding to env.step.")
            else:
                 logger.debug(f"Tool '{tool_name}' is not 'find_flights' or argument count check not applicable. Proceeding to env.step.")

        # If the action was not an intercepted tool call for this specific task type, proceed with the default behavior.
        # Also handles cases where action_match is None (e.g., 'Answer: ...')
        else:
             logger.debug(f"Action format not matched or not intercepted for task type 3. Proceeding to env.step.")

    # --- Logic specific to Task Type 4 (web_browsing) --- (Added based on Analysis Result 5)
    elif task_type_idx == 4:
        # Check for the specific incorrect format: Action: click_url('<url>') End Action
        # This regex captures the tool name 'click_url' followed immediately by parentheses containing a quoted string.
        # It uses re.DOTALL in case the URL contains characters that might be interpreted as newlines by '.'
        incorrect_format_match = re.match(r"Action:\s*click_url\s*\(['\"](.*?)['\"]\)\s*End Action", agent_action, re.IGNORECASE | re.DOTALL)

        if incorrect_format_match:
            intended_url = incorrect_format_match.group(1)
            # Provide specific feedback based on Analysis Result 5
            obs = (f"Feedback: Invalid tool invocation format. You used '{agent_action}'. "
                   f"The tool 'click_url' requires the URL as a comma-separated argument. "
                   f"To click the URL '{intended_url}', the correct format is: Action: click_url, {intended_url} End Action")
            logger.debug(f"Intercepted incorrect 'click_url' format for task type 4. Agent used: '{agent_action}'. Correct format hint provided. Feedback: {obs}")
            return obs, 0.0, False # Return feedback, no reward, task not done
        else:
            # If the specific incorrect format wasn't matched, proceed to the default handler.
            # The standard parser below might still fail if the format is generally invalid,
            # or env.step might handle other errors.
            logger.debug(f"Action format not intercepted for specific 'click_url(<url>)' error for task type 4. Proceeding to default handling.")


    # --- Default behavior ---
    # Execute the action in the environment if it wasn't intercepted above,
    # or if the task is not one of the specific tasks being checked.
    logger.debug(f"Calling env.step for action: {agent_action}")
    try:
        # Ensure env.step is only called once after all checks pass for the current action
        obs, reward, done = env.step(agent_action)
        logger.debug(f"env.step returned: obs='{obs}', reward={reward}, done={done}")
    except Exception as e:
        logger.error(f"Error during env.step execution for action '{agent_action}': {e}", exc_info=True)
        # Provide generic error feedback to the agent to prevent crashes
        # Check if the error message indicates a tool not found, potentially related to the parsing issue
        error_str = str(e)
        if "Could not find tool with name" in error_str and "click_url(" in error_str:
             # Provide more specific feedback if the error looks like the one from Analysis Result 5
             obs = (f"An error occurred: {error_str}. This might indicate an incorrect tool invocation format. "
                    f"Remember to separate the tool name and arguments with a comma. "
                    f"For example: 'Action: click_url, /some/url End Action'.")
        else:
            # Generic error for other exceptions
            obs = f"An error occurred while processing your action: {e}. Please check your action format and arguments."

        reward = 0.0
        done = False # Keep the task running if possible, unless the error indicates otherwise

    return obs, reward, done
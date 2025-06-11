import ast
import logging
import re # Added for hex validation
import string # Added for numeric validation
from typing import Tuple, Dict, Any

# Assume logger is configured elsewhere
logger = logging.getLogger()
# Set a default level if not configured, e.g., logging.INFO or logging.DEBUG
if not logger.hasHandlers():
    logging.basicConfig(level=logging.DEBUG) # Or INFO for less verbosity

# --- Helper Mappings and Functions ---

# Mapping from task_type_idx to task type name
TASK_TYPE_MAP: Dict[int, str] = {
    0: 'message_decoder',
    1: 'cryptobotanists_plant_dna_sequencer',
    2: 'trade_calculator',
    3: 'travel_itinerary_planning',
    4: 'web_browsing',
}

# --- Refined get_task_type_idx_from_name function ---
def get_task_type_idx_from_name(task_name: str, logger: logging.Logger) -> int:
    """
    Helper function to determine task_type_idx from task_name.
    Uses case-insensitive substring matching for all task types based on TASK_TYPE_MAP.
    """
    cleaned_task_name_lower = task_name.strip().lower()
    logger.debug(f"Attempting to match task_name: '{task_name}' (processed as: '{cleaned_task_name_lower}')")

    # Iterate through the canonical task type names and check for substring match
    for idx, type_name in TASK_TYPE_MAP.items():
        type_name_lower = type_name.lower()
        # Use more specific matching if needed, e.g., checking if the name *starts* with the type name
        # For now, substring matching seems sufficient based on provided examples.
        # Use replace to handle potential underscores vs spaces differences if needed
        if type_name_lower.replace('_', ' ') in cleaned_task_name_lower.replace('_', ' '):
            logger.debug(f"Matched task_name '{task_name}' to type index {idx} based on substring '{type_name_lower}' (case-insensitive)")
            return idx

    logger.warning(f"Could not determine task_type_idx for task_name: '{task_name}'. Please check TASK_TYPE_MAP and matching logic. Returning -1.")
    return -1 # Indicate unknown task type


# --- Core Environment Logic Functions ---

def InferRules(task_name: str, task_type_idx: int) -> str:
    """
    Contains the rules for environment and task execute logic for different task types.
    Provides specific rules based on the task type index.
    Refined based on Analysis Results 1, 2, 3, 4, 5, 6, 7, and 8.
    (No changes needed based on the latest refinement request)
    """
    task_type = TASK_TYPE_MAP.get(task_type_idx)
    logger = logging.getLogger() # Obtain logger instance
    logger.debug(f"Getting environment rule for task_name='{task_name}', task_type_idx={task_type_idx} (type: {task_type})")

    if task_type == 'message_decoder':
        # Rule specific to the Message Decoder task, clarifying tool sequence. (Analysis Result 8)
        return """Environment Rules for Message Decoder:
1. Use the provided tools to decode the message according to the instruction.
2. The decoding process requires tools to be used in a specific sequence:
   a. First, use 'convert_hex_to_ascii' to convert the initial hexadecimal string into an ASCII string.
   b. Second, use 'reverse_string' on the ASCII string obtained from the previous step.
   c. Third, use 'decode_caesar_cipher' on the reversed ASCII string obtained from the previous step.
3. Each tool expects input in the format produced by the preceding step. Using tools out of order or with the wrong input type (e.g., trying to reverse the hex string directly) will result in an error.
4. When providing the final decoded message, use the 'Answer:' prefix."""
    elif task_type == 'cryptobotanists_plant_dna_sequencer':
        # Rule specific to the DNA sequencer task, clarifying answer formats.
        return """Environment Rules for DNA Sequencer:
1. Use the provided tools to analyze the DNA sequence according to the instruction.
2. When providing the final answer, use the 'Answer:' prefix.
3. The required format for the final answer depends on the specific instruction. Pay close attention to the instruction details:
   - For tasks asking to count nucleotides or determine frequency, the answer must be strictly in the format of a Python dictionary string. Example: Answer: {'T': 1, 'A': 2, 'C': 2, 'G': 1}
   - For tasks asking to find or identify a sequence (e.g., "find the longest valid DNA sequence"), the answer must be the sequence itself as a string composed only of 'A', 'C', 'G', 'T' characters.
     - IMPORTANT: If asked for the "longest" sequence, provide only the *single* longest valid sequence. Do not concatenate multiple sequences. Example: Answer: AGCTAG
   Natural language sentences, lists, or other formats are not accepted for the final answer."""
    elif task_type == 'travel_itinerary_planning':
        # Rule specific to the Travel Itinerary Planning task, clarifying find_flights usage and answer format.
        return """Environment Rules for Travel Itinerary Planning:
1. Use the provided tools to find flights, hotels, etc., according to the instruction.
2. The 'find_flights' tool requires exactly three arguments in this order: from_location (string), to_location (string), and date (string, format YYYY-MM-DD).
   Example: Action: find_flights, CityA, CityB, 2024-01-01 End Action
3. When providing the final answer (e.g., total budget, number of flights), use the 'Answer:' prefix followed *only* by the final numeric value represented as a string. Do not include explanations, sentences, currency symbols, or units in the answer string itself.
   Example for a budget calculation: Answer: 920
   Example for a count: Answer: 5"""
    elif task_type == 'web_browsing':
        # Rule specific to the Web Browsing task, clarifying action format. (Analysis Result 6)
        return """Environment Rules for Web Browsing:
1. Use the provided tools (like click_url, view, scroll_down, etc.) to navigate and interact with the web page according to the instruction.
2. Tool actions MUST follow the format 'Action: tool_name, argument_1, argument_2, ... End Action'.
3. If a tool takes no arguments, use the format 'Action: tool_name End Action'.
4. IMPORTANT: Do NOT use parentheses '()' in your action string. Arguments are separated by commas only.
   - Correct example with argument: Action: click_url, /about_us End Action
   - Incorrect example with argument: Action: click_url('/about_us') End Action
   - Correct example with no arguments: Action: view End Action
   - Incorrect example with no arguments: Action: view() End Action
5. When providing the final answer, use the 'Answer:' prefix."""
    # Add elif blocks here for other task types if rules are identified later
    # elif task_type == 'trade_calculator': # Example placeholder
    #     return "Rules for trade_calculator..."
    else:
        # Default message if no specific rules are defined for the task type
        logger.debug(f"No specific rules defined for task type '{task_type}' (index {task_type_idx}). Returning default message.")
        return "There are no specific environment rules beyond the standard tool usage instructions provided."

def WrapStep(env: Any, task_name: str, instruction: str, agent_action: str, logger: logging.Logger) -> Tuple[str, float, bool]:
    """
    Process the agent action and return the next observation, reward, and done status.
    Intercepts actions for specific tasks to validate format/logic and provide clearer feedback.
    Refined based on Analysis Results 1-8 and subsequent refinement for message_decoder reward=0 cases.
    Uses refined get_task_type_idx_from_name.
    """
    obs, reward, done = "", 0.0, False

    # Determine task type index from task name using the refined helper function
    task_type_idx = get_task_type_idx_from_name(task_name, logger)

    # Check task types
    is_message_decoder_task = (task_type_idx == 0)
    is_dna_sequencer_task = (task_type_idx == 1)
    is_travel_planning_task = (task_type_idx == 3)
    is_web_browsing_task = (task_type_idx == 4)

    # Intercept 'Answer:' actions specifically for the DNA sequencer task
    if agent_action.startswith("Answer:") and is_dna_sequencer_task:
        logger.debug(f"Processing 'Answer:' action for DNA sequencer task (Task Type 1): {agent_action}")
        answer_content = agent_action[len("Answer:"):].strip()
        instruction_lower = instruction.lower() # Use lower case for case-insensitive matching

        # Determine expected format based on instruction keywords
        expects_dictionary = False
        expects_string = False

        # Keywords suggesting dictionary format (for counts/frequencies)
        dict_keywords = ["count", "frequency", "nucleotide", "distribution"]
        # Expanded keywords suggesting string format (for finding/identifying sequences)
        string_keywords = [
            "find the longest valid dna sequence", "longest valid dna sequence",
            "find the longest sequence", "find sequence", "longest sequence",
            "valid sequence", "extract sequence", "identify sequence",
            "what sequence", "which sequence"
        ]
        # Specific keyword indicating the single longest sequence is expected
        longest_keyword_present = "longest" in instruction_lower

        if any(keyword in instruction_lower for keyword in dict_keywords):
            expects_dictionary = True
            logger.debug(f"Instruction analysis suggests DICTIONARY format is expected for task: '{instruction}'")
        elif any(keyword in instruction_lower for keyword in string_keywords):
            expects_string = True
            logger.debug(f"Instruction analysis suggests STRING format is expected for task: '{instruction}' based on keywords: {string_keywords}")
            if longest_keyword_present:
                 logger.debug("Instruction contains 'longest', implying single longest sequence expected.")
        else:
            logger.warning(f"Could not determine expected answer format (dict/string) from instruction for DNA sequencer task. Instruction: '{instruction}'. Passing action directly to env.step for evaluation.")
            try:
                obs, reward, done = env.step(agent_action)
                logger.debug(f"env.step result (ambiguous format determination): obs='{obs}', reward={reward}, done={done}")
            except Exception as e:
                logger.error(f"Error during env.step for action '{agent_action}' (ambiguous format): {e}", exc_info=True)
                obs = f"An error occurred while processing your action: {e}. Please check your action format and arguments."
                reward = 0.0
                done = False # Allow retry
            return obs, reward, done

        # --- Format Validation Logic for DNA Sequencer ---
        if expects_dictionary:
            logger.debug(f"Validating answer content '{answer_content}' against expected dictionary format.")
            try:
                parsed_value = ast.literal_eval(answer_content)
                if isinstance(parsed_value, dict):
                    logger.debug(f"Answer format is a valid dictionary: {parsed_value}. Proceeding with env.step.")
                    obs, reward, done = env.step(agent_action)
                    logger.debug(f"env.step result for correctly formatted dictionary answer: obs='{obs}', reward={reward}, done={done}")
                else:
                    logger.warning(f"Answer format is incorrect. Expected dictionary, but parsed value is not a dictionary: type={type(parsed_value).__name__}, value='{parsed_value}'")
                    obs = f"Error: The final answer format is incorrect. Based on the instruction, the answer should be a Python dictionary string (e.g., {{'T': 1, 'A': 2}}). Your answer was recognized but it is a {type(parsed_value).__name__}, not a dictionary. Your output was: '{answer_content}'"
                    reward = 0.0
                    done = False
                    logger.debug(f"Returning specific feedback for incorrect type (expected dict): obs='{obs}', reward={reward}, done={done}")
            except (ValueError, SyntaxError, TypeError) as e:
                logger.warning(f"Answer format is incorrect. Expected dictionary, but failed to parse as Python literal. Content: '{answer_content}'. Error: {e}")
                obs = f"Error: The final answer format is incorrect. Based on the instruction, the answer should be a Python dictionary string (e.g., {{'T': 1, 'A': 2}}). Your output could not be parsed as a dictionary. Your output was: '{answer_content}'"
                reward = 0.0
                done = False
                logger.debug(f"Returning specific feedback for parsing failure (expected dict): obs='{obs}', reward={reward}, done={done}")

        elif expects_string:
            logger.debug(f"Validating answer content '{answer_content}' against expected string format.")
            is_dict_like = answer_content.startswith('{') and answer_content.endswith('}')
            is_list_like = answer_content.startswith('[') and answer_content.endswith(']')
            contains_space = ' ' in answer_content
            is_empty = not answer_content
            # Basic check for valid DNA characters (case-insensitive)
            is_plausible_dna = all(c in 'ACGT' for c in answer_content.upper()) if not is_empty else False
            is_valid_string_format = not is_empty and not is_dict_like and not is_list_like and not contains_space and is_plausible_dna

            if is_valid_string_format:
                logger.debug(f"Answer format appears to be a valid sequence string: '{answer_content}'. Proceeding with env.step.")
                try:
                    obs, reward, done = env.step(agent_action)
                    logger.debug(f"env.step result for formatted string answer: obs='{obs}', reward={reward}, done={done}")

                    # --- Check for Concatenation Error (Analysis Result 4) ---
                    incorrect_prefix = "Incorrect! The expected output is: "
                    if reward == 0.0 and obs.startswith(incorrect_prefix) and longest_keyword_present:
                        logger.debug(f"Answer was incorrect (reward=0). Checking for potential concatenation error as 'longest' keyword was present.")
                        try:
                            expected_output = obs[len(incorrect_prefix):].strip()
                            if expected_output.startswith("'") and expected_output.endswith("'"):
                                expected_output = expected_output[1:-1]
                            elif expected_output.startswith('"') and expected_output.endswith('"'):
                                expected_output = expected_output[1:-1]

                            logger.debug(f"Extracted expected output from obs: '{expected_output}'")
                            logger.debug(f"Agent's answer content: '{answer_content}'")

                            if len(answer_content) > len(expected_output) and expected_output in answer_content:
                                logger.warning(f"Detected potential concatenation error. Agent answer '{answer_content}' is longer than expected '{expected_output}' and contains it.")
                                obs = f"Error: The answer format is a valid string, but the content is incorrect. The instruction asked for the *single* longest valid DNA sequence. It seems you may have provided a concatenation of multiple sequences or included invalid parts. The expected output is the single longest sequence. Your output was: '{answer_content}'"
                                reward = 0.0
                                done = False
                                logger.debug(f"Overwrote obs with specific concatenation error feedback: '{obs}'")
                            else:
                                logger.debug("Incorrect answer, but does not appear to be the specific concatenation error pattern.")
                        except Exception as e_parse:
                            logger.error(f"Error parsing expected output from obs ('{obs}') for concatenation check: {e_parse}", exc_info=True)
                            logger.debug("Falling back to original obs from env.step due to parsing error.")

                except Exception as e:
                    logger.error(f"Error during env.step for string answer action '{agent_action}': {e}", exc_info=True)
                    obs = f"An error occurred while processing your answer action '{agent_action}': {e}."
                    reward = 0.0
                    done = False
                    logger.debug(f"Returning error feedback from env.step exception (string answer): obs='{obs}', reward={reward}, done={done}")

            else: # Handle invalid string format
                if answer_content.lower() == "none": # Analysis Result 7
                    logger.warning(f"Answer format is incorrect. Expected sequence string, but received 'None'. Content: '{answer_content}'")
                    obs = f"Error: The final answer format is incorrect. You provided 'None'. Based on the instruction, the answer must be the sequence itself as a string (e.g., AGCTAG), composed only of 'A', 'C', 'G', 'T' characters. 'None' is not a valid DNA sequence string. If no valid sequence was found, 'None' is not the expected format. Please provide the sequence string if one exists, or reconsider the format for indicating no result (e.g., an empty string might be appropriate, though the specific format for 'no sequence' is not explicitly defined)."
                else: # Existing logic for other invalid string formats
                    error_reason = "it is empty" if is_empty else \
                                   "it resembles a dictionary" if is_dict_like else \
                                   "it resembles a list" if is_list_like else \
                                   "it contains spaces (expected a sequence string without spaces)" if contains_space else \
                                   "it contains invalid characters (expected only A, C, G, T)" if not is_plausible_dna else \
                                   "it is not in the expected sequence string format" # Fallback reason
                    logger.warning(f"Answer format is incorrect. Expected sequence string, but {error_reason}. Content: '{answer_content}'")
                    if answer_content == '{}': # Specific feedback for Analysis Result 2
                         obs = f"Error: The final answer format is incorrect. Based on the instruction, the answer must be the sequence itself as a string (e.g., AGCTAG), not an empty dictionary. Your output was: '{answer_content}'"
                    else:
                         obs = f"Error: The final answer format is incorrect. Based on the instruction, the answer must be the sequence itself as a string (e.g., AGCTAG). Your output ({error_reason}) was: '{answer_content}'"

                reward = 0.0
                done = False
                logger.debug(f"Returning specific feedback for incorrect format (expected string): obs='{obs}', reward={reward}, done={done}")

    # Handle Tool Actions
    elif agent_action.startswith("Action:"):
        logger.debug(f"Processing Tool action (Task Type {task_type_idx}): {agent_action}")

        # First, check for the basic ' End Action' structure
        if not agent_action.endswith(" End Action"):
            logger.warning(f"Malformed tool action string (missing ' End Action'): '{agent_action}'")
            obs = f"Error: Your tool action is malformed. It must end with ' End Action'. Your output was: '{agent_action}'"
            reward = 0.0
            done = False
            logger.debug(f"Returning feedback for malformed tool action: obs='{obs}', reward={reward}, done={done}")
            return obs, reward, done # Return early, do not proceed

        # Extract content between 'Action:' and ' End Action'
        action_content = agent_action[len("Action:"): -len(" End Action")].strip()
        parts = [p.strip() for p in action_content.split(',')]
        tool_name = parts[0] if parts else ""
        tool_args = parts[1:] if len(parts) > 1 else []

        # --- Specific Handling for Message Decoder Task (Task Type 0) - Analysis Result 8 & Refinement ---
        if is_message_decoder_task:
            logger.debug(f"Applying specific validation for Message Decoder task (Task Type 0). Tool: '{tool_name}', Args: {tool_args}")
            try:
                # Attempt to execute the action
                obs, reward, done = env.step(agent_action)
                logger.debug(f"env.step initial result for Message Decoder action '{tool_name}': obs='{obs}', reward={reward}, done={done}")

                # --- START REFINEMENT for reward=0.0 cases ---
                specific_feedback_provided = False
                if reward == 0.0:
                    # Check for specific misuse patterns first
                    if tool_name == "reverse_string" and tool_args:
                        input_arg = tool_args[0]
                        # Check if input looks like hex
                        if re.match(r'^[0-9a-fA-F]+$', input_arg):
                            logger.warning(f"Message Decoder tool 'reverse_string' failed (reward=0) and input '{input_arg}' looks like hex. Providing specific feedback.")
                            obs = f"Error executing tool 'reverse_string'. This tool expects a standard text string as input, not a hexadecimal string. Did you remember to convert the hex string to ASCII using 'convert_hex_to_ascii' first? The required sequence is Hex -> ASCII -> Reverse -> Caesar Decode."
                            done = False
                            specific_feedback_provided = True
                            logger.debug(f"Overwrote obs with specific feedback for reversing hex: '{obs}'")

                    elif tool_name == "decode_caesar_cipher" and tool_args:
                         input_arg = tool_args[0]
                         # Check if input looks like hex (less common, but possible if agent skips both steps)
                         if re.match(r'^[0-9a-fA-F]+$', input_arg):
                             logger.warning(f"Message Decoder tool 'decode_caesar_cipher' failed (reward=0) and input '{input_arg}' looks like hex. Providing specific feedback.")
                             obs = f"Error executing tool 'decode_caesar_cipher'. This tool expects a standard text string (usually the reversed ASCII string) as input, not a hexadecimal string. It seems you might have missed both the 'convert_hex_to_ascii' and 'reverse_string' steps. The required sequence is Hex -> ASCII -> Reverse -> Caesar Decode."
                             done = False
                             specific_feedback_provided = True
                             logger.debug(f"Overwrote obs with specific feedback for decoding hex: '{obs}'")
                         # Optional: Could add check here if input looks like non-reversed ASCII if needed

                    # If no specific pattern matched, check for generic failure messages in obs
                    if not specific_feedback_provided:
                        generic_failure_indicators = ["Failed to execute tool", "Error executing tool", "Could not process input"]
                        is_generic_failure = any(indicator in obs for indicator in generic_failure_indicators)
                        if is_generic_failure:
                            logger.warning(f"Message Decoder tool '{tool_name}' executed but failed logically (reward=0). Original Obs: '{obs}'. Providing semi-specific feedback.")
                            error_prefix = f"Error executing tool '{tool_name}'"
                            if tool_name == "reverse_string":
                                obs = f"{error_prefix}. This tool expects a standard text string as input. It seems the input might not be in the correct format (e.g., still hexadecimal or incorrect type). Did you remember to convert the hex string to ASCII using 'convert_hex_to_ascii' first? The required sequence is Hex -> ASCII -> Reverse -> Caesar Decode."
                            elif tool_name == "decode_caesar_cipher":
                                obs = f"{error_prefix}. This tool expects a standard text string (usually the reversed ASCII string) as input. It seems the input might not be in the correct format. Did you remember to reverse the ASCII string using 'reverse_string' first? The required sequence is Hex -> ASCII -> Reverse -> Caesar Decode."
                            elif tool_name == "convert_hex_to_ascii":
                                obs = f"{error_prefix}. This tool expects a hexadecimal string as input. Please ensure the input provided is a valid hex string and that you haven't already converted it. The required sequence is Hex -> ASCII -> Reverse -> Caesar Decode."
                            else:
                                obs = f"{error_prefix}. The tool failed, likely due to incorrect input type or order. Please check the tool's expected input and the required sequence: Hex -> ASCII -> Reverse -> Caesar Decode. Original observation: {obs}"
                            done = False
                            logger.debug(f"Overwrote obs with semi-specific feedback for logical failure: '{obs}'")
                        # else: # reward is 0 but no specific pattern or generic failure message found
                        #    logger.debug(f"Tool '{tool_name}' returned reward=0 but obs '{obs}' doesn't indicate a known failure pattern. Passing through.")

                # --- END REFINEMENT ---

            except Exception as e:
                # Handle cases where env.step itself crashes
                logger.error(f"Exception during env.step for Message Decoder action '{agent_action}': {e}", exc_info=True)
                # Provide specific feedback based on the tool that failed
                error_prefix = f"Error executing tool '{tool_name}'"
                if tool_name == "reverse_string":
                    obs = f"{error_prefix}. This tool expects a standard text string as input. An exception occurred, possibly because the input was not in the correct format (e.g., still hexadecimal). Did you remember to convert the hex string to ASCII using 'convert_hex_to_ascii' first? The required sequence is Hex -> ASCII -> Reverse -> Caesar Decode. Original error: {e}"
                elif tool_name == "decode_caesar_cipher":
                    obs = f"{error_prefix}. This tool expects a standard text string (usually the reversed ASCII string) as input. An exception occurred, possibly because the input was not in the correct format. Did you remember to reverse the ASCII string using 'reverse_string' first? The required sequence is Hex -> ASCII -> Reverse -> Caesar Decode. Original error: {e}"
                elif tool_name == "convert_hex_to_ascii":
                    obs = f"{error_prefix}. This tool expects a hexadecimal string as input. An exception occurred, possibly because the input was not a valid hex string. The required sequence is Hex -> ASCII -> Reverse -> Caesar Decode. Original error: {e}"
                else:
                    # Generic fallback for other potential tools or errors in this task
                    obs = f"{error_prefix}. An exception occurred: {e}. Please check the tool's expected input and the required sequence: Hex -> ASCII -> Reverse -> Caesar Decode."

                reward = 0.0
                done = False # Allow the agent to retry
                logger.debug(f"Returning specific feedback for Message Decoder tool exception: obs='{obs}', reward={reward}, done={done}")

        # --- Specific Handling for Web Browsing Task (Task Type 4) - Analysis Result 6 ---
        elif is_web_browsing_task:
            logger.debug("Applying specific validation for Web Browsing task (Task Type 4).")
            # Check for invalid parentheses
            if '(' in action_content or ')' in action_content:
                logger.warning(f"Invalid format for Web Browsing tool action: contains parentheses. Action content: '{action_content}'")
                obs = f"Error: Invalid tool action format for Web Browsing. Do not use parentheses '()'. Use the format 'Action: tool_name, argument1, argument2 End Action' or 'Action: tool_name End Action' with comma-separated arguments. Your action was: '{agent_action}'"
                reward = 0.0
                done = False
                logger.debug(f"Returning specific feedback for parentheses error (Web Browsing): obs='{obs}', reward={reward}, done={done}")
                # Return early, do not call env.step
            else:
                # Format seems okay (no parentheses), proceed to env.step
                logger.debug(f"Web Browsing action format appears valid (no parentheses). Passing to env.step: '{agent_action}'")
                try:
                    obs, reward, done = env.step(agent_action)
                    logger.debug(f"env.step result for Web Browsing action: obs='{obs}', reward={reward}, done={done}")
                except Exception as e:
                    logger.error(f"Error during env.step for Web Browsing action '{agent_action}': {e}", exc_info=True)
                    obs = f"An error occurred while executing the Web Browsing action '{agent_action}': {e}. Please check tool name and arguments."
                    reward = 0.0
                    done = False
                    logger.debug(f"Returning error feedback from env.step exception (Web Browsing): obs='{obs}', reward={reward}, done={done}")

        # --- Specific Handling for Travel Planning Task (Task Type 3) ---
        elif is_travel_planning_task:
            logger.debug("Applying specific validation for Travel Planning task (Task Type 3).")
            # Note: tool_name and tool_args parsed earlier

            if tool_name == "find_flights":
                expected_arg_count = 3
                if len(tool_args) == expected_arg_count:
                    logger.debug(f"Correct number of arguments ({len(tool_args)}) provided for find_flights. Proceeding with env.step.")
                    try:
                        obs, reward, done = env.step(agent_action)
                        logger.debug(f"env.step result for find_flights: obs='{obs}', reward={reward}, done={done}")
                    except Exception as e:
                        logger.error(f"Error during env.step for find_flights action '{agent_action}': {e}", exc_info=True)
                        obs = f"An error occurred while executing find_flights with args {tuple(tool_args)}: {e}. Please check argument values and format (e.g., date YYYY-MM-DD)."
                        reward = 0.0
                        done = False
                        logger.debug(f"Returning error feedback from env.step exception (find_flights): obs='{obs}', reward={reward}, done={done}")
                else: # Incorrect number of arguments for find_flights (Analysis Result 3)
                    logger.warning(f"Incorrect number of arguments for find_flights. Expected {expected_arg_count}, got {len(tool_args)}. Args received: {tool_args}")
                    obs = f"Error: Incorrect arguments for tool 'find_flights'. Expected {expected_arg_count} arguments: from_location (string), to_location (string), date (string, YYYY-MM-DD). You provided {len(tool_args)} arguments: {tool_args}. Example: Action: find_flights, CityA, CityB, 2024-01-01 End Action"
                    reward = 0.0
                    done = False
                    logger.debug(f"Returning specific feedback for incorrect find_flights args: obs='{obs}', reward={reward}, done={done}")
                    # Do NOT call env.step here
            else:
                # Handle other tools within travel planning - pass through
                logger.debug(f"Tool '{tool_name}' is not 'find_flights' or validation not implemented. Passing action to env.step.")
                try:
                    obs, reward, done = env.step(agent_action)
                    logger.debug(f"env.step result for other tool '{tool_name}': obs='{obs}', reward={reward}, done={done}")
                except Exception as e:
                    logger.error(f"Error during env.step for action '{agent_action}': {e}", exc_info=True)
                    obs = f"An error occurred while processing your action '{agent_action}': {e}. Please check your action format and arguments."
                    reward = 0.0
                    done = False
                    logger.debug(f"Returning error feedback from env.step exception: obs='{obs}', reward={reward}, done={done}")

        # --- General Handling for Tool Actions in other tasks ---
        else:
            logger.debug(f"Passing Tool action for other task (Task Type {task_type_idx}) directly to env.step.")
            try:
                obs, reward, done = env.step(agent_action)
                logger.debug(f"env.step result: obs='{obs}', reward={reward}, done={done}")
            except Exception as e:
                logger.error(f"Error during env.step for action '{agent_action}': {e}", exc_info=True)
                obs = f"An error occurred while processing your action '{agent_action}': {e}. Please check your action format and arguments."
                reward = 0.0
                done = False # Allow retry if it was an execution error
                logger.debug(f"Returning error feedback from env.step exception: obs='{obs}', reward={reward}, done={done}")

    # Handle Answers for non-DNA tasks
    elif agent_action.startswith("Answer:"):
        logger.debug(f"Processing 'Answer:' action (Task Type {task_type_idx}): {agent_action}.")

        # --- Specific Handling for Travel Planning Task (Task Type 3) - Analysis Result 5 ---
        if is_travel_planning_task:
            logger.debug("Applying specific answer format validation for Travel Planning task (Task Type 3).")
            answer_content = agent_action[len("Answer:"):].strip()

            is_valid_numeric_string = False
            if answer_content: # Check if not empty
                try:
                    # Attempt to parse as float, but also check for letters to disallow things like '123 dollars'
                    float(answer_content)
                    if any(c in string.ascii_letters for c in answer_content):
                         logger.warning(f"Answer content '{answer_content}' contains letters, considered non-numeric string for this task.")
                         is_valid_numeric_string = False
                    else:
                         is_valid_numeric_string = True
                         logger.debug(f"Answer content '{answer_content}' is considered a valid numeric string.")
                except ValueError:
                    logger.warning(f"Answer content '{answer_content}' could not be parsed as a number.")
                    is_valid_numeric_string = False
            else:
                logger.warning("Answer content is empty.")
                is_valid_numeric_string = False

            if is_valid_numeric_string:
                logger.debug(f"Answer format is a valid numeric string: '{answer_content}'. Proceeding with env.step.")
                try:
                    obs, reward, done = env.step(agent_action)
                    logger.debug(f"env.step result for numeric answer (Travel Task): obs='{obs}', reward={reward}, done={done}")
                except Exception as e:
                    logger.error(f"Error during env.step for numeric answer action '{agent_action}' (Travel Task): {e}", exc_info=True)
                    obs = f"An error occurred while processing your numeric answer action '{agent_action}': {e}."
                    reward = 0.0
                    done = False
                    logger.debug(f"Returning error feedback from env.step exception (numeric answer, Travel Task): obs='{obs}', reward={reward}, done={done}")
            else:
                logger.warning(f"Answer format is incorrect for Travel Planning task. Expected numeric string, got: '{answer_content}'")
                obs = f"Error: The final answer format is incorrect. For this task, the answer should be the calculated numeric value only, represented as a string (e.g., '920' or '15.5'). Your output must not contain explanations, sentences, currency symbols, or units. Your output was: '{answer_content}'"
                reward = 0.0
                done = False
                logger.debug(f"Returning specific feedback for incorrect format (expected numeric string, Travel Task): obs='{obs}', reward={reward}, done={done}")
                # Do NOT call env.step here

        # --- Handling for other non-DNA/non-Travel tasks ---
        else:
            logger.debug(f"Passing 'Answer:' action for non-DNA/non-Travel/non-MessageDecoder task (Task Type {task_type_idx}) directly to env.step.")
            try:
                obs, reward, done = env.step(agent_action)
                logger.debug(f"env.step result: obs='{obs}', reward={reward}, done={done}")
            except Exception as e:
                logger.error(f"Error during env.step for answer action '{agent_action}': {e}", exc_info=True)
                obs = f"An error occurred while processing your answer '{agent_action}': {e}."
                reward = 0.0
                done = False # Allow retry if appropriate
                logger.debug(f"Returning error feedback from env.step exception: obs='{obs}', reward={reward}, done={done}")

    # Handle cases where the agent output is malformed (not starting with Action: or Answer:)
    else:
        logger.warning(f"Agent output is not in the expected 'Answer:' or 'Action:' format: '{agent_action}'. Treating as invalid action.")
        obs = f"Error: Your output must start with either 'Answer:' for the final answer or 'Action:' to use a tool. Your output was: '{agent_action}'"
        reward = 0.0
        done = False
        logger.debug(f"Returning feedback for invalid action format: obs='{obs}', reward={reward}, done={done}")
        # Do not call env.step for fundamentally malformed actions

    # Return the observation, reward, and done status
    return obs, reward, done
import re
import logging
import sys
from typing import List, Tuple, Any # Added for type hinting if needed

# Assuming logger is configured elsewhere and passed into WrapStep
# Example configuration:
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
# logger = logging.getLogger(__name__)

# Placeholder for the environment class if needed for type hinting or checks
# class SingleAlfredTWEnv:
#     # Dummy implementation for testing structure
#     _state = {} # Simple state tracking for testing
#     def step(self, action):
#         action_str = action[0]
#         print(f"Simulating env.step('{action_str}') with state: {self._state}")
#         obs, reward, done, info = "Okay.", 0.0, False, {'won': [False]}
#
#         # Simplified simulation logic for testing
#         if action_str == "go to cabinet 1":
#             if self._state.get('location') != 'cabinet 1':
#                 obs = "You arrive at cabinet 1. On the cabinet 1, you see nothing."
#                 self._state['location'] = 'cabinet 1'
#             else:
#                 obs = "Nothing happens." # Already there
#         elif action_str == "go to fridge 1":
#             if self._state.get('location') != 'fridge 1':
#                 obs = "You arrive at fridge 1. The fridge 1 is closed."
#                 self._state['location'] = 'fridge 1'
#             else:
#                 obs = "Nothing happens." # Already there
#         elif action_str == "go to countertop 1":
#             if self._state.get('location') != 'countertop 1':
#                 obs = "You arrive at countertop 1. On the countertop 1, you see nothing."
#                 self._state['location'] = 'countertop 1'
#             else:
#                 obs = "Nothing happens." # Already there
#         elif action_str == "go to desklamp 1": # Example for Analysis Result 4
#             obs = "Nothing happens." # Cannot go to small object
#         elif action_str == "examine cabinet 1":
#             if self._state.get('location') == 'cabinet 1':
#                 obs = "On the cabinet 1, you see nothing."
#             else:
#                 obs = "Nothing happens." # Too far
#         elif action_str == "open fridge 1":
#              if self._state.get('location') == 'fridge 1':
#                  obs = "You open the fridge 1. Inside the fridge 1, you see a bowl 1."
#                  self._state['fridge 1 open'] = True
#              else:
#                  obs = "Nothing happens." # Too far
#         elif action_str == "take bowl 1 from fridge 1":
#              if self._state.get('location') == 'fridge 1' and self._state.get('fridge 1 open'):
#                  obs = "You take the bowl 1 from the fridge 1."
#                  self._state['inventory'] = 'bowl 1'
#              elif self._state.get('location') != 'fridge 1':
#                   obs = "Nothing happens." # Too far
#              elif not self._state.get('fridge 1 open'):
#                   obs = "Nothing happens." # Closed
#              else:
#                   obs = "Nothing happens." # Other reason
#         elif action_str == "cool bowl 1 with fridge 1":
#              if self._state.get('location') == 'fridge 1' and self._state.get('inventory') == 'bowl 1':
#                   obs = "You cool the bowl 1 with the fridge 1."
#              else:
#                   obs = "Nothing happens." # Not holding or not at location
#         elif action_str == "move bowl 1 to countertop 1":
#              if self._state.get('location') == 'countertop 1' and self._state.get('inventory') == 'bowl 1':
#                   obs = "You put the bowl 1 on the countertop 1."
#                   self._state['inventory'] = None
#                   self._state['bowl_location'] = 'countertop 1'
#              else:
#                   obs = "Nothing happens." # Not holding or not at location
#         else:
#             # Default response for unhandled actions
#             obs = "Okay."
#
#         print(f"-> New state: {self._state}, Obs: '{obs}'")
#         # Ensure return format matches expected structure
#         return [obs], float(info['won'][0]), [False], {'won': [info['won'][0]]} # Adjusted return to match expected processing
#
#     def reset(self):
#         self._state = {'location': 'start', 'inventory': None, 'fridge 1 open': False, 'bowl_location': 'fridge 1'}
#         # Example init_obs format
#         init_obs_text = """You are in the middle of a room. Looking quickly around you, you see a cabinet 1, a fridge 1, a countertop 1, and a shelf 1. On the shelf 1, you see a desklamp 1.
# Your task is to cool a bowl and put it on the counter."""
#         return [init_obs_text], {'won': [False]}
#     def init_env(self, batch_size):
#         return self
# # env = SingleAlfredTWEnv() # Dummy env for testing structure

def InferRules(init_obs: str, task: str) -> str:
    """
    Contains the rules for environment and task execute logic for different task types.
    Reflects rules identified in Analysis Results 1, 2, 3 & 4.
    """
    # Combined rules based on Analysis Result 1, 2, 3 and 4
    rules = [
        "Rule 1: To interact with specific objects or receptacles (e.g., open, close, take from, examine), you often need to be at their location first. Use 'go to [receptacle]' to move closer before interacting.",
        "Rule 2: Actions like 'move [object] to [receptacle]', 'heat', 'cool', 'clean', or 'slice' often require you to be holding the necessary object(s) first (using 'take [object] from [receptacle]') and be at the target receptacle/location.",
        "Rule 3: Ensure receptacles like cabinets or fridges are open before trying to take items from them or put items into them.",
        "Rule 4: The 'go to' action is primarily for moving to larger locations or receptacles (like tables, counters, fridges, cabinets). You cannot use 'go to' to move directly to small objects; instead, go to the receptacle or surface where the object is located." # Added based on Analysis Result 4
    ]
    return "\n".join(rules)

def _parse_receptacles(init_obs: str, logger) -> List[str]:
    """Helper function to parse receptacle names from the initial observation."""
    # Regex to find patterns like "a table 1", "an armchair 2", "a countertop"
    # It looks for 'a' or 'an' followed by words (potentially including spaces) and optionally a number.
    # It captures the full name (e.g., "cabinet 1", "fridge 1", "countertop 1").
    # This regex assumes receptacles are introduced in the format "..., you see a/an [receptacle name]..."
    matches = re.findall(r'\b(?:a|an)\s+([\w\s]+\d*)\b', init_obs)
    # Normalize to lowercase and strip whitespace for consistent matching
    receptacles = [m.strip().lower() for m in matches]
    logger.debug(f"Parsed receptacles from init_obs: {receptacles}")
    return receptacles

def WrapStep(env: Any, init_obs: str, task: str, agent_action: str, logger: logging.Logger) -> Tuple[str, bool, bool]:
    """
    Process the agent action:
    1. Parse the agent action string.
    2. Check for specific invalid action patterns ("put...in", "task_complete") and provide feedback without executing (Refinement for Analysis Result 3).
    3. If action pattern is valid, execute the action using env.step.
    4. Check for specific failure cases (like interactions from afar or without prerequisites)
       indicated by "Nothing happens." and provide clearer feedback (Refinement for Analysis Result 1, 2 & 4).
    5. Return the next observation, reward (as bool), and done status (as bool).
    """
    action_item = {
        'matched': False,
        'action': None,
        'object': None,
        'receptacle': None,
        'object2': None, # For slice
        'raw_target': None # Store the primary target (receptacle or object)
    }

    original_agent_action = agent_action # Keep original for logging/debugging if needed
    agent_action_normalized = agent_action.lower().strip() # Normalize action

    # --- Action Parsing Logic ---
    # (Existing parsing logic remains unchanged)
    # Simple actions without parameters
    if agent_action_normalized == 'look' or agent_action_normalized == 'inventory':
        action_item['matched'] = True
        action_item['action'] = agent_action_normalized

    # Pattern: go to (receptacle)
    elif agent_action_normalized.startswith('go to '):
        receptacle = agent_action_normalized[6:].strip()
        if receptacle:
            action_item['matched'] = True
            action_item['action'] = 'go to'
            action_item['receptacle'] = receptacle # Store normalized target
            action_item['raw_target'] = receptacle # Store normalized target

    # Pattern: open/close (receptacle)
    elif agent_action_normalized.startswith('open ') or agent_action_normalized.startswith('close '):
        action_verb = agent_action_normalized.split(' ', 1)[0]
        receptacle = agent_action_normalized[len(action_verb):].strip()
        if receptacle:
            action_item['matched'] = True
            action_item['action'] = action_verb
            action_item['receptacle'] = receptacle
            action_item['raw_target'] = receptacle

    # Pattern: take (object) from (receptacle)
    elif agent_action_normalized.startswith('take ') and ' from ' in agent_action_normalized:
        parts = agent_action_normalized.split(' from ', 1)
        obj = parts[0][5:].strip()
        receptacle = parts[1].strip()
        if obj and receptacle:
            action_item['matched'] = True
            action_item['action'] = 'take from' # Using 'take from' as action key
            action_item['object'] = obj
            action_item['receptacle'] = receptacle
            action_item['raw_target'] = receptacle # Target location

    # Pattern: move (object) to (receptacle)
    elif agent_action_normalized.startswith('move ') and ' to ' in agent_action_normalized:
        parts = agent_action_normalized.split(' to ', 1)
        obj = parts[0][5:].strip() # 'move ' is 5 chars
        receptacle = parts[1].strip()
        if obj and receptacle:
            action_item['matched'] = True
            action_item['action'] = 'move to' # Using 'move to' as action key
            action_item['object'] = obj
            action_item['receptacle'] = receptacle
            action_item['raw_target'] = receptacle # Target location

    # Pattern: examine (something)
    elif agent_action_normalized.startswith('examine '):
        something = agent_action_normalized[8:].strip()
        if something:
            action_item['matched'] = True
            action_item['action'] = 'examine'
            action_item['raw_target'] = something

    # Pattern: use (object)
    elif agent_action_normalized.startswith('use '):
        obj = agent_action_normalized[4:].strip()
        if obj:
            action_item['matched'] = True
            action_item['action'] = 'use'
            action_item['object'] = obj
            action_item['raw_target'] = obj

    # Pattern: heat/clean/cool (object) with (receptacle)
    elif (agent_action_normalized.startswith('heat ') or agent_action_normalized.startswith('clean ') or agent_action_normalized.startswith('cool ')) and ' with ' in agent_action_normalized:
        parts = agent_action_normalized.split(' with ', 1)
        action_verb_obj = parts[0]
        action_verb = action_verb_obj.split(' ', 1)[0]
        obj = action_verb_obj[len(action_verb):].strip()
        receptacle = parts[1].strip()
        if obj and receptacle:
            action_item['matched'] = True
            action_item['action'] = action_verb
            action_item['object'] = obj
            action_item['receptacle'] = receptacle
            action_item['raw_target'] = receptacle # Target is the instrument/location

    # Pattern: slice (object) with (object)
    elif agent_action_normalized.startswith('slice ') and ' with ' in agent_action_normalized:
        parts = agent_action_normalized.split(' with ', 1)
        obj = parts[0][6:].strip() # 'slice ' is 6 chars
        obj2 = parts[1].strip() # The tool
        if obj and obj2:
            action_item['matched'] = True
            action_item['action'] = 'slice'
            action_item['object'] = obj
            action_item['object2'] = obj2
            action_item['raw_target'] = obj # Target is the object being sliced

    # --- End Action Parsing Logic ---


    # --- Refinement based on Analysis Result 3 ---
    # Check for specific invalid action patterns BEFORE attempting execution
    if not action_item['matched']:
        # Check for "put ... in ..." pattern
        if agent_action_normalized.startswith('put ') and ' in ' in agent_action_normalized:
            try:
                parts = agent_action_normalized.split(' in ', 1)
                obj = parts[0][4:].strip() # 'put ' is 4 chars
                receptacle = parts[1].strip()
                if obj and receptacle:
                    feedback = f"The action '{original_agent_action}' is not recognized. Did you mean 'move {obj} to {receptacle}'?"
                    logger.debug(f"Caught 'put...in' pattern. Suggesting 'move...to'. Action: '{original_agent_action}'")
                    # Return feedback without executing, reward=False, done=False
                    return feedback, False, False
            except Exception as e:
                logger.warning(f"Error parsing 'put...in' pattern for action '{original_agent_action}': {e}")
                # Fall through to generic unrecognized action feedback

        # Check for "task_complete" action
        elif agent_action_normalized == 'task_complete':
            feedback = "There is no explicit 'task_complete' action. Please continue interacting with the environment using the available actions until the task is marked as complete by the environment."
            logger.debug(f"Caught 'task_complete' action. Providing feedback.")
            # Return feedback without executing, reward=False, done=False
            return feedback, False, False

        # Handle other unrecognized actions
        else:
            feedback = f"The action '{original_agent_action}' is not recognized or supported by the environment. Please use one of the documented action formats (e.g., 'go to ...', 'take ... from ...', 'move ... to ...', etc.)."
            logger.warning(f"Agent action '{original_agent_action}' did not match any known patterns and is not a handled invalid pattern. Providing generic feedback.")
            # Return feedback without executing, reward=False, done=False
            return feedback, False, False

    # --- End Refinement for Analysis Result 3 ---


    # If the action matched a known pattern, proceed to execute it
    action_to_execute = original_agent_action # Use the original case-sensitive action string for the env

    # Execute the action in the environment
    logger.debug(f"Executing matched action in env: {action_to_execute}")
    try:
        # Use env.step exactly as specified
        obs_list, reward_float, done_list, info = env.step([action_to_execute])
        # Extract results as per required format
        obs, reward_bool, done_bool = obs_list[0], info['won'][0], done_list[0]
        logger.debug(f"Raw observation from env: {obs}")
        logger.debug(f"Raw reward from env: {reward_bool}")
        logger.debug(f"Raw done from env: {done_bool}")

    except Exception as e:
        logger.error(f"Error during env.step execution for action '{action_to_execute}': {e}")
        # Provide error feedback to the agent
        obs = f"An error occurred while trying to execute the action: {action_to_execute}. Please try a different action."
        reward_bool = False
        done_bool = False
        # Optionally re-raise or handle specific exceptions if needed
        return obs, bool(reward_bool), bool(done_bool)


    # --- Refinement based on Analysis Result 1, 2 & 4 (Handling "Nothing happens.") ---
    # Check if the action resulted in "Nothing happens." and provide clearer feedback.
    # This check should only apply if the action was successfully executed (i.e., matched a pattern and env.step ran)
    if obs == "Nothing happens." and action_item['matched']:
        original_obs = obs # Keep original obs for logging
        action_verb = action_item['action']
        target = action_item['raw_target'] # Use normalized target from parsing
        obj = action_item['object']
        receptacle = action_item['receptacle'] # Use normalized receptacle from parsing
        tool = action_item['object2'] # For slice

        new_obs = original_obs # Default to original if no specific rule applies

        # Feedback tailored to why "Nothing happens." might occur for VALID actions

        # Analysis Result 4: 'go to' failure
        if action_verb == 'go to' and target:
            # Parse receptacles from init_obs to check if the target is valid for 'go to'
            known_receptacles = _parse_receptacles(init_obs, logger)
            # Check if the normalized target is in the list of normalized known receptacles
            if target not in known_receptacles:
                # Target is likely a small object, not a valid 'go to' destination
                new_obs = f"You cannot move directly to objects like '{target}'. Try using 'go to' with a larger location like a table, counter, or shelf where the object might be."
                logger.debug(f"Agent tried 'go to' non-receptacle '{target}'. Providing specific feedback.")
            else:
                # Target is a known receptacle, but 'go to' still failed
                new_obs = f"Could not move to the {target}. You might already be there or there might be an issue moving there. Try 'look' to confirm your location or try moving elsewhere first."
                logger.debug(f"Agent 'go to' receptacle '{target}' resulted in 'Nothing happens'. Providing generic failure feedback.")

        # Analysis Result 1: Examine/Open/Close from afar
        elif action_verb in ['examine', 'open', 'close'] and target:
            new_obs = f"You might be too far away to {action_verb} the {target}. Try moving closer first, for example using 'go to {target}'."

        # Analysis Result 2: Take prerequisites
        elif action_verb == 'take from' and obj and receptacle:
            # Could be too far, receptacle closed, or object not there
            new_obs = f"Could not take the {obj} from the {receptacle}. Make sure you are at the {receptacle}, it is open, and the {obj} is inside."

        # Analysis Result 2: Move prerequisites
        elif action_verb == 'move to' and obj and receptacle:
            # Could be not holding object, or not at target receptacle
            new_obs = f"Could not move the {obj} to the {receptacle}. Make sure you are holding the {obj} (use 'take {obj} from ...' first) and are at the {receptacle} (use 'go to {receptacle}' first)."

        # Analysis Result 2: Heat/Cool/Clean prerequisites
        elif action_verb in ['heat', 'cool', 'clean'] and obj and receptacle:
            preposition = "with" # Common preposition for these actions
            # Could be not holding object, or not at target receptacle
            new_obs = f"Could not {action_verb} the {obj} {preposition} the {receptacle}. Make sure you are holding the {obj} and are at the {receptacle}."

        # Analysis Result 2: Slice prerequisites
        elif action_verb == 'slice' and obj and tool:
             # Could be not holding tool, or not near object to be sliced
             new_obs = f"Could not slice the {obj} with the {tool}. Make sure you are holding the {tool} (the slicing instrument) and are near the {obj}."

        # Add more specific checks if needed for other actions like 'use'

        if new_obs != original_obs:
            obs = new_obs
            logger.debug(f"Modified 'Nothing happens.' observation for failed '{action_verb}' action. Original: '{original_obs}', New: '{obs}'")
        else:
            # Log if "Nothing happens." occurred but wasn't handled by specific rules above
            # This case should be less frequent now with the 'go to' handling added.
            logger.debug(f"'Nothing happens.' observed for action '{action_to_execute}', but no specific feedback rule matched. Action details: {action_item}")

    # --- End Refinement for "Nothing happens." ---

    # Ensure return types match the required signature: obs (str), reward (bool), done (bool)
    # The environment wrapper seems to return reward as bool via info['won'][0]
    # and done as bool via done[0]
    return obs, bool(reward_bool), bool(done_bool)
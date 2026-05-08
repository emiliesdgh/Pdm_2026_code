import ollama
import threading
import json
from ig_logger import setup_logger

# logger = setup_logger("gesture_runtime_log2.txt")  # Initialize the logger

class LLMInferenceAgent:
    # Mixtral requires 24GB to 32GB RAM
    # if the computer doesn't have enough RAM, it might crash
    # ==> in this case, use Mistral
    # before using, run in terminal: ollama pull mixtral
    # --> this allows to download the model
    # def __init__(self, model_name="mixtral"): # too slow and not necessarily better

    # to use if mixtral, is too heavy & makes the video feed lag or crash the script
    # if using Mistral instead, make sure to run 'ollama pull mistral' in the terminal first to download the model
    def __init__(self, model_name="mistral"):  
        self.model_name = model_name
        self.is_inferencing = False
        # self.current_prediction = "Waiting for hand gesture..."

        self.current_prediction = "Waiting..."
        self.current_reasoning = "Waiting..."

    def analyze_gesture_async(self, symbolic_str):
        """
        Starts a background thread to ask Ollama what gesture is.
        """
        if self.is_inferencing:
            return  # Don't start a new request is one is currently running
        
        self.is_inferencing = True
        thread = threading.Thread(target=self._query_ollama, args=(symbolic_str,))
        thread.start()


    def _query_ollama(self, symbolic_str):
        """
        Function that send the prompt to ollama and get the response, runs in a separate thread.
        the system prompt is defined here to avoid re-defining it every time in the main loop, and to have a single source of truth for the system prompt.
        it gives the context to the LLM about the task and what it should do with the symbolic string that is sent from the main loop.
        """
        
        # # 75.3% accuracy with the system_prompt below
        # system_prompt = (
        #     "You are the visual reasoning cortex for an autonomous robot. Your task is to map the user's kinematic hand state to ONE of four intents: "
        #     "[PICK_UP, NAVIGATE_THERE, STOP, SEARCH_AREA].\n\n"
            
        #     "STEP 1: IDENTIFY THE HAND POSE\n"
        #     "- Pointing Pose: Index finger is straight, while Middle, Ring, and Pinky are bent. (CRITICAL: In this pose, ignore what the Thumb is doing or touching. Thumb contact is natural when pointing and does NOT mean grabbing).\n"
        #     "- Fist Pose: All fingers are bent.\n"
        #     "- Open Palm Pose: All fingers are straight.\n\n"

        #     "STEP 2: MAP POSE + MOTION TO INTENT (STRICT RULES)\n"
        #     "Use these exact rules to determine the intent:\n\n"
            
        #     "Rule for NAVIGATE_THERE:\n"
        #     "- If the hand is in a Pointing Pose AND is mostly Stationary, the intent is NAVIGATE_THERE.\n"
        #     "- If the hand is in an Open Palm Pose AND the palm is facing Down AND is Stationary, the intent is NAVIGATE_THERE (indicating a flat path).\n\n"

        #     "Rule for SEARCH_AREA:\n"
        #     "- If the motion is 'Oscillating Left & Right' OR 'Hand Rotation', the intent is ALMOST ALWAYS SEARCH_AREA. This applies whether the hand is in an Open Palm Pose or a Pointing Pose (e.g., pointing around the room).\n\n"

        #     "Rule for PICK_UP:\n"
        #     "- If the motion is 'Bending Fingers' or 'Hand Open/Close', the intent is PICK_UP (active grabbing).\n"
        #     "- If the hand is in a Fist Pose AND has a Linear Translation motion (e.g., Up, Down, Left, Right), the intent is PICK_UP (moving a grabbed object).\n"
        #     "- If the hand is NOT Pointing, and the Thumb is in contact with multiple fingertips, it is a pinch/grab, meaning PICK_UP.\n\n"

        #     "Rule for STOP:\n"
        #     "- If the hand is in a Fist Pose AND is strictly 'Stationary', the intent is STOP.\n"
        #     "- If the hand is in an Open Palm Pose AND is strictly 'Stationary' AND the palm is facing Inward or Outward, the intent is STOP.\n\n"

        #     "Output ONLY a valid JSON object with exactly two keys: 'intent' (one of the 4 commands) and 'reasoning' (a brief explanation of how you applied the rules above). Do not output any markdown formatting or extra text outside the JSON."
        # )

        # # 66.6% accuracy with the system_promt below
        # system_prompt = (
        #     "You are the visual reasoning cortex for an autonomous robot. Your task is to map the user's kinematic hand state to ONE of four intents: "
        #     "[PICK_UP, NAVIGATE_THERE, STOP, SEARCH_AREA].\n\n"
            
        #     "STEP 1: IDENTIFY THE HAND POSE\n"
        #     "- Pointing Pose: Index finger is straight, while Middle, Ring, and Pinky are bent. (CRITICAL: In this pose, IGNORE what the Thumb is doing or touching (the Thumb could be straight or bent). Thumb contact, especially only to the Middle fingertip, is natural when pointing and does NOT mean grabbing).\n"
        #     "- Fist Pose: All fingers (especially Index, Middle, Ring, Pinky) are bent, the Thumb might be extended. \n"# or Index, Middle, Ring and Pinky are bent (the Thumb might be extended).\n"
        #     "- Open Palm Pose: All fingers are straight.\n\n"

        #     "STEP 2: MAP MOTION + POSE TO INTENT (STRICT RULES)\n"
        #     "Use these exact rules to determine the intent:\n\n"
            
        #     "Rule for NAVIGATE_THERE:\n"
        #     "- If the hand is Stationary AND in a Pointing Pose, the intent is NAVIGATE_THERE.\n"
        #     "- If the hand is in a Pointing Pose and has a 'Slow...' motion (e.g. 'Slow Bending Fingers'), ignore the motion and consider it as NAVIGATE_THERE, because it is likely just camera jitter while pointing.\n\n"

        #     "Rule for SEARCH_AREA:\n"
        #     "- If the motion is 'Oscillating Left & Right' OR 'Hand Rotation', the intent is ALMOST ALWAYS SEARCH_AREA. This applies whether the hand is in an Open Palm Pose or a Pointing Pose (e.g., pointing around the room), but NOT in a Fist Pose.\n\n"

        #     "Rule for PICK_UP:\n"
        #     "- If the motion is different from 'Stationary' AND the hand pose is different from 'Open Palm' AND the rules below apply, the intent is PICK_UP. \n" #(Motion can override pose for PICK_UP, but not for NAVIGATE_THERE. For example, if the hand is Pointing but has a 'Bending Fingers' motion, it is likely a quick grab while pointing, so PICK_UP overrides NAVIGATE_THERE in this case.)
        #     "- If the motion is 'Bending Fingers' or 'Hand Open/Close', the intent is PICK_UP (active grabbing).\n"
        #     "- If the hand is in a Fist pose AND the motion is different from 'Stationary'.\n"
        #     "- If the hand is in a Fist Pose AND has a Linear Translation motion (e.g., Up, Down, Left, Right), the intent is PICK_UP (moving a grabbed object).\n"
        #     "- If the hand is NOT Pointing, and the Thumb is in contact with MULTIPLE fingertips, it is a pinch/grab, meaning PICK_UP.\n\n"

        #     "Rule for STOP:\n"
        #     "- If the hand is Stationary AND in a Fist Pose, the intent is STOP.\n"
        #     "- If the hand is in an Open Palm Pose AND is strictly 'Stationary' AND the palm is strictly facing Inward or Outward, the intent is STOP.\n\n"

        #     "Output ONLY a valid JSON object with exactly two keys: 'intent' (one of the 4 commands) and 'reasoning' (a brief explanation of how you applied the rules above). Do not output any markdown formatting or extra text outside the JSON."
        # )

        # XX% accuracy with the system prompt below
        system_prompt = (
            "You are the visual reasoning cortex for an autonomous robot. Your task is to interpret a user's free-form hand gesture and map it to ONE of four intents: "
            "[PICK_UP, NAVIGATE_THERE, STOP, SEARCH_AREA].\n\n"
            
            "To understand the user's intent, analyze the physical metaphor of their hand pose and motion:\n\n"

            "1. NAVIGATE_THERE (Metaphor: Directing or Pathing)\n"
            "- Look for the Index finger being straight (Pointing). If the user is pointing, they are directing the robot. Ignore what the thumb is doing.\n"
            "- Alternatively, look for a flat Open Palm (all fingers straight) facing Down and held Stationary, representing a flat path.\n\n"

            "2. SEARCH_AREA (Metaphor: Scanning or Exploring)\n"
            "- The defining feature of searching is the motion. Look for 'Oscillating Left & Right', 'Hand Rotation', or a sweeping 'Linear Translation' with an Open Palm.\n"
            "- This motion overrides most hand poses, as users scan the room differently.\n\n"

            "3. PICK_UP (Metaphor: Grabbing, Pinching, or Lifting)\n"
            "- Look for 'Bending Fingers' or 'Hand Open/Close' (the act of grasping).\n"
            "- Look at Thumb Contact: If the Thumb is in contact with exactly one 'fingertip' (singular), it is a precise pinch. If it is in contact with multiple 'fingertips' (plural), it is a full grab. Both mean PICK_UP.\n"
            "- Look for a Fist (all fingers bent) moving with a 'Linear Translation' (mimicking carrying an object).\n\n"

            "4. STOP (Metaphor: Blocking or Halting)\n"
            "- Look for rigid, Stationary poses intended to halt action.\n"
            "- This is usually an Open Palm facing Outward/Inward (like a stop sign), or a tight, Stationary Fist.\n\n"

            "STEP-BY-STEP REASONING REQUIRED:\n"
            "1. What is the physical shape of the hand (Pointing, Flat, Fist, Pinching)?\n"
            "2. What is the motion doing (Scanning, Lifting, Halting)?\n"
            "3. Which of the 4 metaphors does this combination best fit?\n\n"

            "Output ONLY a valid JSON object with exactly two keys: 'intent' (one of the 4 commands) and 'reasoning' (your step-by-step logic). Do not output markdown formatting."
        )

        
        try:
            response =ollama.chat(model=self.model_name, messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': symbolic_str}
            ], format='json', options={'temperature': 0.1})#, options = {'num_predict': 20}) # num_predict limits the number of tokens generated by the model --> can be suffocating for the llm
                                        # temp = 0.1 makes the output more deterministic, less "creative" & forces to be more analytical & follow the rules
            #, format='json', options={'temperature': 0.1, 'num_predict': 100})   # other option to try 
            
            response_text = response['message']['content'].strip()
            # Parse the string into a Python Dictionary
            try:
                prediction_json = json.loads(response_text)
                self.current_intent = prediction_json.get("intent", "UNKNOWN")
                self.current_reasoning = prediction_json.get("reasoning", "No reasoning.")
                
                print(f"\n[NEW INTENT DECODED]: {self.current_intent}")
                print(f"[REASONING]: {self.current_reasoning}\n")

                # logger.info(f"[NEW INTENT DECODED]: {self.current_intent}")
                # logger.info(f"[REASONING]: {self.current_reasoning}")
                return prediction_json
            
            except json.JSONDecodeError:
                print(f"❌ JSON Parse Error. Raw Output: {response_text}")
                # logger.error(f"❌ JSON Parse Error. Raw Output: {response_text}")
                # return {"intent": "UNKNOWN"}
                self.current_intent = "UNKNOWN"
                return {"intent": "UNKNOWN", "reasoning": "Failed to parse LLM response."}

        except Exception as e:
            print(f"❌ Ollama Error: {e}")
            # logger.error(f"❌ Ollama Error: {e}")
            # return {"intent": "UNKNOWN"}
            self.current_intent = "UNKNOWN"
            return {"intent": "UNKNOWN", "reasoning": f"Ollama Error: {e}"}
        finally:
            self.is_inferencing = False
        
        #     self.current_prediction = response['message']['content'].strip()
        #     print(f"\n[LLM Prediction]: {self.current_prediction}\n")
        # except Exception as e:
        #     print(f"Ollama Error: {e}")
        # finally:
        #     self.is_inferencing = False


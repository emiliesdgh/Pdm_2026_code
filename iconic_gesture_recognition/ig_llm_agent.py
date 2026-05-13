import ollama
import threading
import json
from ig_logger import setup_logger

# logger = setup_logger("gesture_runtime_log2.txt")  # Initialize the logger

class LLMInferenceAgent:
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
        
        # # 75.3% -- 74% now (13.05.2026) accuracy with the system_prompt below
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

        # XX % accuracy with the system_prompt below
        system_prompt = (
            "You are the visual reasoning cortex for an autonomous robot. Your task is to interpret a user's free-form hand gesture and map it to ONE of four intents: "
            "[PICK_UP, NAVIGATE_THERE, STOP, SEARCH_AREA].\n\n"

            "IDENTIFY THE HAND POSE\n"
            "- Pointing Pose: Index finger is straight, while Middle, Ring, and Pinky are bent. (CRITICAL: In this pose, ignore what the Thumb is doing or touching. Thumb contact is natural when pointing and does NOT mean grabbing).\n"
            "- Fist Pose: All fingers are bent.\n"
            "- Open Palm Pose: All fingers are straight.\n\n"
            
            "To understand the user's intent, analyze the physical metaphor of their hand pose combined with their TEMPORAL MOTION LOG:\n"
            "1. NAVIGATE_THERE: Pointing (Index straight) or Flat Open Palm facing Down. Spatial Trajectory is Stationary. \n"# or a smooth Linear Translation.\n"
            "2. SEARCH_AREA: Spatial Trajectory shows scanning motions ('Oscillating', 'Hand Rotation', or sweeping 'Linear Translation'). Hand Articulation is 'None' or 'Pointing' in motion different from Stationary.\n"
            "3. PICK_UP: Look for 'Bending Fingers' or 'All fingers bending towards the palm' in the Hand Articulation. This is often combined with a 'Linear Translation' in the Spatial Trajectory (mimicking lifting an object). Fingertips going into contact with the Thumb tip is also a strong indicator.\n"
            "4. STOP: Rigid poses (Open Palm facing Outward/Inward or a Fist). Both Spatial Trajectory and Hand Articulation MUST be 'Stationary' and 'None'.\n\n"

            "STEP-BY-STEP REASONING REQUIRED:\n"
            "1. Identify the hand pose based on finger flexion states.\n"
            "2. Match the Hand State and Temporal Motions to an intent.\n"
            "3. Look at the ROBOT VISION data. Does the hand position logically align with an object in the scene? If so, identify the 'target'. If no object is relevant, the target is 'None'.\n"
            "4. Assess your confidence (0.0 to 1.0). Give > 0.8 for clear matches, and < 0.6 if the gesture is ambiguous.\n\n"

            "Output ONLY a valid JSON object with exactly four keys: 'intent', 'target', 'confidence_score' (float), and 'reasoning'. Do not output markdown formatting."
        )

        
        try:
            response =ollama.chat(model=self.model_name, messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': symbolic_str}
            ], format='json', options={'temperature': 0.1})

            response_text = response['message']['content'].strip()
            # Parse the string into a Python Dictionary
            try:
                prediction_json = json.loads(response_text)
                self.current_intent = prediction_json.get("intent", "UNKNOWN")
                self.current_reasoning = prediction_json.get("reasoning", "No reasoning.")
                
                print(f"\n[NEW INTENT DECODED]: {self.current_intent}")
                print(f"[REASONING]: {self.current_reasoning}\n")

                return prediction_json
            
            except json.JSONDecodeError:
                print(f"❌ JSON Parse Error. Raw Output: {response_text}")
                self.current_intent = "UNKNOWN"
                return {"intent": "UNKNOWN", "reasoning": "Failed to parse LLM response."}

        except Exception as e:
            print(f"❌ Ollama Error: {e}")
            self.current_intent = "UNKNOWN"
            return {"intent": "UNKNOWN", "reasoning": f"Ollama Error: {e}"}
        finally:
            self.is_inferencing = False



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
        
               
        # # 74% Accuracy with system prompt below (13.05.2026) with gesture_dataset_good
        # system_prompt = (
        #     "You are the visual reasoning cortex for an autonomous robot. Your task is to map the user's kinematic hand state to ONE of four intents: "
        #     "[PICK_UP, NAVIGATE_THERE, STOP, SEARCH_AREA].\n\n"
            
        #     "STEP 1: IDENTIFY THE HAND POSE AND MOTIONS\n"
        #     "Define the user's state using these strict categories:\n"
        #     "- Pointing Pose: Index finger is straight. (Thumb state is irrelevant here).\n"
        #     "- Fist Pose: All fingers are bent.\n"
        #     "- Open Palm Pose: All fingers are straight.\n"
        #     "- Pinching Pose: Thumb is explicitly in contact with one or more fingertips.\n\n"

        #     "STEP 2: MAP TO INTENT (APPLY IN ORDER)\n"
        #     "Match the Step 1 definitions and the Temporal Motion Log to determine intent:\n\n"
            
        #     "SEARCH_AREA (Scanning):\n"
        #     "- If Spatial Motion contains 'Oscillating', 'Waving', or 'Hand Rotation', the intent is SEARCH_AREA. (This motion overrides all hand poses).\n"
        #     "- If the pose is Open Palm AND Spatial Motion is a 'Linear Translation', the intent is SEARCH_AREA.\n\n"

        #     "PICK_UP (Grabbing / Lifting):\n"
        #     "- If Articulation contains 'Closing' or 'Pinching', the intent is PICK_UP.\n"
        #     "- If the pose is Pinching AND Spatial Motion is 'Stationary', the intent is PICK_UP. (Exception: If the hand is in a Pointing Pose, do NOT trigger this).\n"
        #     "- If the pose is Fist AND Spatial Motion is a 'Linear Translation', the intent is PICK_UP.\n\n"

        #     "NAVIGATE_THERE (Directing):\n"
        #     "- If the pose is Pointing AND Spatial Motion is 'Stationary' or a 'Linear Translation', the intent is NAVIGATE_THERE.\n"
        #     "- If the pose is Open Palm AND Spatial Motion is 'Stationary' AND the palm faces 'Down', the intent is NAVIGATE_THERE.\n\n"

        #     "STOP (Halting):\n"
        #     "- If the pose is Fist AND Spatial Motion is 'Stationary' (with NO thumb contact), the intent is STOP.\n"
        #     "- If the pose is Open Palm AND Spatial Motion is 'Stationary' AND the palm faces 'Inward' or 'Outward', the intent is STOP.\n\n"

        #     "STEP 3: ENVIRONMENT TARGETING\n"
        #     "Look at the ROBOT VISION data. If the user's Spatial Motion or Pointing direction aligns with a detected object, that object is the 'target'. If no object aligns, target is 'None'.\n\n"

        #     "Output ONLY a valid JSON object with exactly four keys: 'intent', 'target', 'confidence_score' (float 0.0 to 1.0), and 'reasoning' (explain your Step 1 and Step 2 logic)."
        # )

        # xx% Accuracy with system prompt below (13.05.2026) with gesture_dataset_good
        system_prompt = (
            "You are the visual reasoning cortex for an autonomous robot. Your task is to map the user's kinematic hand state to ONE of four intents: "
            "[PICK_UP, NAVIGATE_THERE, STOP, SEARCH_AREA].\n\n"
            
            "STEP 1: IDENTIFY THE TRUE HAND POSE\n"
            "Define the user's state using these mutually exclusive categories:\n"
            "- Pointing Pose: Index finger is straight. (CRITICAL: If the Index is straight, it is ALWAYS Pointing. Ignore any thumb contact).\n"
            "- Fist Pose: All fingers are bent.\n"
            "- Open Palm Pose: All fingers are straight.\n"
            "- Pinching Pose: Thumb is in contact with one or more fingertips. (CRITICAL: Do NOT classify as Pinching if the hand is in a Pointing Pose or a Fist Pose).\n\n"

            "STEP 2: MAP TO INTENT (APPLY IN EXACT ORDER)\n"
            "Match the Step 1 definitions and the Temporal Motion Log to determine intent:\n\n"
            
            "1. THE GRAB OVERRIDE (Intent: PICK_UP):\n"
            "- If Articulation contains 'Closing' or 'Grabbing', the intent is ALWAYS PICK_UP. (This overrides all other motions like Rotation or Oscillating).\n\n"

            "2. SEARCH_AREA (Scanning):\n"
            "- If Spatial Motion contains 'Oscillating', 'Waving', or 'Hand Rotation', the intent is SEARCH_AREA.\n"
            "- If the pose is Open Palm AND Spatial Motion is a 'Linear Translation', the intent is SEARCH_AREA.\n\n"

            "3. PICK_UP (Stationary or Carrying):\n"
            "- If the pose is Pinching AND Spatial Motion is 'Stationary', the intent is PICK_UP.\n"
            "- If the pose is Fist AND Spatial Motion is a 'Linear Translation', the intent is PICK_UP.\n\n"

            "4. NAVIGATE_THERE (Directing):\n"
            "- If the pose is Pointing AND Spatial Motion is 'Stationary' or a 'Linear Translation', the intent is NAVIGATE_THERE.\n"
            "- If the pose is Open Palm AND Spatial Motion is 'Stationary' AND the palm faces 'Down', the intent is NAVIGATE_THERE.\n\n"

            "5. STOP (Halting):\n"
            "- If the pose is Fist AND Spatial Motion is 'Stationary', the intent is STOP.\n"
            "- If the pose is Open Palm AND Spatial Motion is 'Stationary' AND the palm faces 'Inward' or 'Outward', the intent is STOP.\n\n"

            "STEP 3: ENVIRONMENT TARGETING\n"
            "Look at the ROBOT VISION data. If the user's Spatial Motion or Pointing direction aligns with a detected object, that object is the 'target'. If no object aligns, target is 'None'.\n\n"

            "Output ONLY a valid JSON object with exactly four keys: 'intent', 'target', 'confidence_score' (float 0.0 to 1.0), and 'reasoning' (explain your Step 1 and Step 2 logic)."
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



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

        self.current_intent = "Waiting..."
        self.current_target = "None"
        self.current_confidence = 0.0

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
        # # 89% Accuracy with system prompt below (13.05.2026) with gesture_dataset_v2
        # system_prompt = (
        #     "You are the reasoning cortex for an autonomous robot. Map the user's kinematic hand state to ONE of four intents: "
        #     "[PICK_UP, NAVIGATE_THERE, STOP, SEARCH_AREA].\n\n"
            
        #     "STEP 1: IDENTIFY THE TRUE HAND POSE\n"
        #     "Define the user's state using these mutually exclusive categories:\n"
        #     "- Pointing Pose: Index finger is straight or Index finger AND Thumb are straight (CRITICAL: If the Index is straight, it is ALWAYS Pointing. Ignore any thumb contact).\n"
        #     "- Open Palm Pose: All fingers are straight exclusively.\n"
        #     "- Fist Pose: All fingers are bent excusively AND Articulation contains 'None' or 'Opening' or 'Static Fingers'.\n"
        #     "- Pinching Pose: Thumb is in contact with the Index and possible with more fingertips.\n\n"# (CRITICAL: Do NOT classify as Pinching if the hand is in a Pointing Pose or a Fist Pose).\n\n"
        #     # the pinching needs the index in contact with the thumb to be a pinch


        #     "STEP 2: MAP TO INTENT (APPLY IN EXACT ORDER)\n"
        #     "Match the Step 1 definitions and the Temporal Motion Log to determine intent:\n\n"
        #     # "Follow this EXACT logical sequence. The first match determines the intent:\n\n"
            
        #     "1. GRABBING & LIFTING (Intent: PICK_UP):\n"
        #     "- IF Articulation contains 'Closing' or 'Grabbing' or 'Pinching', the intent is ALWAYS PICK_UP. (This overrides all other rules).\n"
        #     "- IF the hand is holding a Pinch (Thumb in contact with Index and possibly other fingertips), the Intent is PICK_UP. (CRITICAL: the Index finger must be in contact with the Thumb)\n"
        #     "- IF the hand is a Fist AND Spatial Motion is a 'Linear Translation', the Intent is PICK_UP.\n\n"

        #     "2. SEARCH_AREA (Scanning):\n"
        #     "- IF Articulation contains 'None' or 'Opening' or 'Static Fingers'.\n"
        #     "- If Spatial Motion contains 'Oscillating', 'Waving', or 'Hand Rotation', the intent is SEARCH_AREA.\n"
        #     "- If the pose is Open Palm AND Spatial Motion is a 'Linear Translation', the intent is SEARCH_AREA.\n\n"

        #     "3. NAVIGATE_THERE (Directing):\n"
        #     "- IF Articulation contains 'None' or 'Opening' or 'Static Fingers'.\n"
        #     "- If the pose is Pointing AND Spatial Motion is 'Stationary' or a 'Linear Translation', the intent is NAVIGATE_THERE.\n"
        #     "- If the pose is Open Palm AND Spatial Motion is 'Stationary' AND the palm orientation is 'Down', the intent is NAVIGATE_THERE.\n\n"

        #     "4. STOP (Halting):\n"
        #     "- IF Articulation contains 'None' or 'Opening' or 'Static Fingers'.\n"
        #     "- If the pose is Fist AND Spatial Motion is 'Stationary', the intent is STOP.\n"
        #     "- If the pose is Open Palm AND Spatial Motion is 'Stationary' AND the palm orientation is 'Inward' or 'Outward', the intent is STOP.\n\n"
        #     # have palm orientation rather than palm facing to avoid hallucination

        #     "STEP 3: ENVIRONMENT TARGETING\n"
        #     "Look at the ROBOT VISION data. If the user's Spatial Motion or Pointing direction aligns with a detected object, that object is the 'target'. If no object aligns, target is 'None'.\n\n"

        #     # "Output ONLY a valid JSON object with exactly four keys: 'intent', 'target', 'confidence_score' (float 0.0 to 1.0), and 'reasoning' (explain your Step 1 and Step 2 logic)."
        #     "You MUST output a valid JSON object strictly matching this structure. Fill out the 'analysis' section FIRST:\n"
        #     "{\n"
        #     "  \"analysis\": {\n"
        #     "    \"articulation_state\": \"(Copy the Articulation from the log)\",\n"
        #     "    \"spatial_motion\": \"(Copy the Spatial Motion from the log)\",\n"
        #     "    \"is_index_straight\": \"true/false\",\n"
        #     "    \"is_thumb_touching\": \"true/false\"\n"
        #     "  },\n"
        #     "  \"intent\": \"(ONE OF THE 4 INTENTS)\",\n"
        #     "  \"target\": \"(Object or None)\",\n"
        #     "  \"confidence_score\": 0.9,\n"
        #     "  \"reasoning\": \"(Brief explanation of the rule matched)\"\n"
        #     "}"
        # )

        # 89% Accuracy with system prompt below (13.05.2026) with gesture_dataset_v2
        system_prompt = (
            "You are the reasoning cortex for an autonomous robot. Map the user's kinematic hand state to ONE of four intents: "
            "[PICK_UP, NAVIGATE_THERE, STOP, SEARCH_AREA].\n\n"
            
            "STEP 1: IDENTIFY THE TRUE HAND POSE\n"
            "Define the user's state using these mutually exclusive categories:\n"
            "- Pointing Pose: Index finger is straight or Index finger AND Thumb are straight (CRITICAL: If the Index is straight, it is ALWAYS Pointing. Ignore any thumb contact).\n"
            "- Open Palm Pose: All fingers are straight exclusively.\n"
            "- Fist Pose: All fingers are bent excusively AND Articulation contains 'None' or 'Opening' or 'Static Fingers'.\n"
            "- Pinching Pose: Thumb is in contact with the Index and possible with more fingertips.\n\n"# (CRITICAL: Do NOT classify as Pinching if the hand is in a Pointing Pose or a Fist Pose).\n\n"
            # the pinching needs the index in contact with the thumb to be a pinch


            "STEP 2: MAP TO INTENT (APPLY IN EXACT ORDER)\n"
            "Match the Step 1 definitions and the Temporal Motion Log to determine intent:\n\n"
            # "Follow this EXACT logical sequence. The first match determines the intent:\n\n"
            
            "1. GRABBING & LIFTING (Intent: PICK_UP):\n"
            "- IF Articulation contains 'Closing' or 'Grabbing' or 'Pinching', the intent is ALWAYS PICK_UP. (This overrides all other rules).\n"
            "- IF the hand is holding a Pinch (Thumb in contact with Index and possibly other fingertips), the Intent is PICK_UP. (CRITICAL: the Index finger must be in contact with the Thumb)\n"
            "- IF the hand is a Fist AND Spatial Motion is a 'Linear Translation', the Intent is PICK_UP.\n\n"

            "2. SEARCH_AREA (Scanning):\n"
            "- IF Articulation contains 'None' or 'Opening' or 'Static Fingers'.\n"
            "- If Spatial Motion contains 'Oscillating', 'Waving', or 'Hand Rotation', the intent is SEARCH_AREA.\n"
            "- If the pose is Open Palm AND Spatial Motion is a 'Linear Translation', the intent is SEARCH_AREA.\n\n"

            "3. NAVIGATE_THERE (Directing):\n"
            "- IF Articulation contains 'None' or 'Opening' or 'Static Fingers'.\n"
            "- If the pose is Pointing AND Spatial Motion is 'Stationary' or a 'Linear Translation', the intent is NAVIGATE_THERE.\n"
            "- If the pose is Open Palm AND Spatial Motion is 'Stationary' AND the palm orientation is 'Down', the intent is NAVIGATE_THERE.\n\n"

            "4. STOP (Halting):\n"
            "- IF Articulation contains 'None' or 'Opening' or 'Static Fingers'.\n"
            "- If the pose is Fist AND Spatial Motion is 'Stationary', the intent is STOP.\n"
            "- If the pose is Open Palm AND Spatial Motion is 'Stationary' AND the palm orientation is 'Inward' or 'Outward', the intent is STOP.\n\n"
            # have palm orientation rather than palm facing to avoid hallucination

            "STEP 3: CONFIDENCE SCORING (CRITICAL)\n"
            "Your final 'confidence_score' MUST be grounded in the Sensor Data Quality.\n"
            "- If Camera Tracking Confidence is low (< 0.75), your final confidence MUST NOT exceed the camera's confidence.\n"
            "- Deduct 0.2 from your confidence if the gesture feels ambiguous or partially matches two rules.\n\n"

            "STEP 4: ENVIRONMENT TARGETING\n"
            "Look at the ROBOT VISION data. If the user's Spatial Motion or Pointing direction aligns with a detected object, that object is the 'target'. If no object aligns, target is 'None'.\n\n"

            # "Output ONLY a valid JSON object with exactly four keys: 'intent', 'target', 'confidence_score' (float 0.0 to 1.0), and 'reasoning' (explain your Step 1 and Step 2 logic)."
            "You MUST output a valid JSON object strictly matching this structure. Fill out the 'analysis' section FIRST:\n"
            "{\n"
            "  \"analysis\": {\n"
            "    \"articulation_state\": \"(Copy the Articulation from the log)\",\n"
            "    \"spatial_motion\": \"(Copy the Spatial Motion from the log)\",\n"
            "    \"is_index_straight\": \"true/false\",\n"
            "    \"is_thumb_touching\": \"true/false\"\n"
            "  },\n"
            "  \"intent\": \"(ONE OF THE 4 INTENTS)\",\n"
            "  \"target\": \"(Object or None)\",\n"
            "  \"confidence_score\": 0.0,\n"
            "  \"reasoning\": \"(Brief explanation of the rule matched)\"\n"
            "}"
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
                self.current_target = prediction_json.get("target", "None")
                self.current_confidence = prediction_json.get("confidence_score", 0.0)
                self.current_reasoning = prediction_json.get("reasoning", "No reasoning.")
                
                print(f"\n[NEW INTENT DECODED]: {self.current_intent} | Target: {self.current_target} | Confidence: {self.current_confidence}\n")
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



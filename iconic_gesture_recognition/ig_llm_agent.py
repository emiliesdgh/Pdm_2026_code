import ollama
import threading
import json
from ig_logger import setup_logger
import time

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
        self.current_latency = 0.0

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
        # # % Accuracy with system prompt below (18.05.2026) w/ given hand poses mathematically
        # # without environmental context

        # system_prompt = (
        #     "You are the reasoning cortex for an autonomous robot. Map the user's kinematic hand state to ONE of four intents: "
        #     "[PICK_UP, NAVIGATE_THERE, STOP, SEARCH_AREA].\n\n"

        #     "STEP 1: IDENTIFY THE HAND POSE\n"
        #     "Check if the hand is a known or unknown pose described in the 'HAND STATE'bloc.\n\n"

        #     "STEP 2: MAP TO INTENT (APPLY IN EXACT ORDER)\n"
        #     "A. GRABBING (PICK_UP):\n"
        #     "- IF Articulation contains 'Closing', 'Grabbing', or 'Pinching' -> Intent is exclusivelyPICK_UP. (Overrides everything).\n"
        #     "- IF the Hand Pose is Pinching AND Index is touching Thumb -> Intent is PICK_UP.\n\n"

        #     "B. SCANNING (SEARCH_AREA):\n"
        #     "- IF Spatial Motion contains 'Oscillating', 'Waving', or 'Rotation' -> Intent is SEARCH_AREA.\n"
        #     "- IF the Hand Pose is Open Palm AND Spatial Motion is 'Linear Translation' -> Intent is SEARCH_AREA.\n\n"

        #     "C. DIRECTING (NAVIGATE_THERE):\n"
        #     "- IF the Hand Pose is Pointing AND Spatial Motion is 'Stationary' or 'Linear Translation' -> Intent is NAVIGATE_THERE.\n"
        #     "- IF the Hand Pose is Open Palm AND Spatial Motion is 'Stationary' AND palm orientation is 'Down' -> Intent is NAVIGATE_THERE.\n\n"

        #     "D. HALTING (STOP):\n"
        #     "- IF the Hand Pose is Fist AND Spatial Motion is 'Stationary' -> Intent is STOP.\n"
        #     "- IF the Hand Pose is Open Palm AND Spatial Motion is 'Stationary' AND palm orientation is 'Inward' or 'Outward' -> Intent is STOP.\n\n"

        #     "STEP 3: OUTPUT FORMAT\n"
        #     "Output ONLY a valid JSON object. Do not add comments. Fill out the 'analysis' section FIRST:\n"
        #     "{\n"
        #     "  \"analysis\": {\n"
        #     "    \"articulation_state\": \"(Copy from log)\",\n"
        #     "    \"spatial_motion\": \"(Copy from log)\"\n"
        #     "    \"determined_pose\": \"(Write Pointing Pose, Open Palm Pose, Fist Pose, or Pinching Pose)\",\n"
        #     "    \"is_grab_override_active\": true/false (MUST BE true IF articulation_state contains 'Closing' or 'Grabbing' or 'Pinching')\,\n"
        #     "    \"final_logic\": \"IF is_grab_override_active is true,Intent MUST be PICK_UP (Explain which rule from Step 2 matched to determine the intent)\"\n"
        #     "  },\n"
        #     "  \"intent\": \"(ONE OF THE 4 INTENTS)\",\n"
        #     "  \"target\": \"(Extract the object name from ROBOT VISION if applicable, otherwise None)\",\n"
        #     "  \"confidence_score\": 0.9,\n"
        #     "  \"reasoning\": \"(Explain based on the final_logic)\"\n"
        #     "}"
        # )

        # system prompt w/ given hand poses mathematically modified to receive environmental context
        # system_prompt = (
        #     "You are the reasoning cortex for an autonomous robot. Map the user's kinematic hand state to ONE of four intents: "
        #     "[PICK_UP, NAVIGATE_THERE, STOP, SEARCH_AREA].\n\n"

        #     "STEP 1: IDENTIFY THE HAND POSE\n"
        #     "Check if the hand is a known or unknown pose described in the 'HAND STATE'bloc.\n\n"

        #     "STEP 2: MAP TO INTENT (APPLY IN EXACT ORDER)\n"
        #     "A. GRABBING (PICK_UP):\n"
        #     "- IF Articulation contains 'Closing', 'Grabbing', or 'Pinching' -> Intent is exclusivelyPICK_UP. (Overrides everything).\n"
        #     "- IF the Hand Pose is Pinching AND Index is touching Thumb -> Intent is PICK_UP.\n\n"

        #     "B. SCANNING (SEARCH_AREA):\n"
        #     "- IF Spatial Motion contains 'Oscillating', 'Waving', or 'Rotation' -> Intent is SEARCH_AREA.\n"
        #     "- IF the Hand Pose is Open Palm AND Spatial Motion is 'Linear Translation' -> Intent is SEARCH_AREA.\n\n"

        #     "C. DIRECTING (NAVIGATE_THERE):\n"
        #     "- IF the Hand Pose is Pointing AND Spatial Motion is 'Stationary' or 'Linear Translation' -> Intent is NAVIGATE_THERE.\n"
        #     "- IF the Hand Pose is Open Palm AND Spatial Motion is 'Stationary' AND palm orientation is 'Down' -> Intent is NAVIGATE_THERE.\n\n"

        #     "D. HALTING (STOP):\n"
        #     "- IF the Hand Pose is Fist AND Spatial Motion is 'Stationary' -> Intent is STOP.\n"
        #     "- IF the Hand Pose is Open Palm AND Spatial Motion is 'Stationary' AND palm orientation is 'Inward' or 'Outward' -> Intent is STOP.\n\n"

        #     "STEP 3: ENVIRONMENTAL FEASIBILITY (CRITICAL SAFETY CHECK)\n"
        #     "Compare the Intent from Step 2 against the 'ROBOT VISION' context:\n"
        #     # "- IF Intent is STOP -> The action is ALWAYS safe and possible, proceed normally. Ignore the vision.\n"
        #     # "- IF Intent is SEARCH_AREA -> The action is ALWAYS safe and possible, proceed normally. Ignore the vision.\n"
        #     # "- IF Intent is PICK_UP but the visionsays 'No objects visible' or 'No box' -> The action is impossible. You MUST output a confidence_score of 0.0.\n"
        #     # "- IF Intent is NAVIGATE_THERE but the vision says 'Path blocked or 'Obstacle in path' -> The action is unsafe. You MUST output a confidence_score of 0.0.\n"
        #     "- IF Intent is PICK_UP but the visions ays 'No objects visible' or 'No box' -> The action is blocked. \n" #You MUST output a confidence_score of 0.0.\n"
        #     "- IF Intent is NAVIGATE_THERE but the vision says 'Obstacle in path' -> The action is blocked.\n" # You MUST output a confidence_score of 0.0.\n"
        #     # "- For all other Intent -> Action is safe and logically possible -> Proceed normally. \n\n"
        #     "- IF Intent is STOP -> The action is ALWAYS safe and possible, proceed normally. \n" 
        #     "- IF Intent is SEARCH_AREA -> The action is ALWAYS safe and possible, proceed normally. \n" 

        #     "STEP 4: OUTPUT FORMAT\n"
        #     "Output ONLY a valid JSON object. Do not add comments. Fill out the 'analysis' section FIRST:\n"
        #     "{\n"
        #     "  \"analysis\": {\n"
        #     "    \"articulation_state\": \"(Copy from log)\",\n"
        #     "    \"spatial_motion\": \"(Copy from log)\"\n"
        #     "    \"determined_pose\": \"(Write Pointing Pose, Open Palm Pose, Fist Pose, or Pinching Pose)\",\n"
        #     "    \"base_intent\": \"(Which intent matched in Step 2)\",\n"

        #     "    \"is_grab_override_active\": true/false (MUST BE true IF articulation_state contains 'Closing' or 'Grabbing' or 'Pinching')\,\n"
        #     "    \"vision_context\": \"(Summarize the ROBOT VISION data)\",\n"
        #     # "    \"does_vision_block_intent\": true/false (MUST be true ONLY IF Step 3 says the action is impossible or unsafe. Otherwise, write false)\",\n"
        #     "    \"is_action_blocked\": true/false (CRITICAL: MUST be false IF base_intent is STOP or SEARCH_AREA)\",\n" # MUST be true ONLY IF Step 3 says the action is blocked)\",\n"
        #     # "    \"is_environment_safe\": true/false (MUST be true IF base_intent is STOP. Otherwise, write false if the vision makes the base_intent unsafe or impossible)\",\n"

        #     "    \"final_logic\": \"IF is_grab_override_active is true, Intent MUST be PICK_UP (Explain which rule from Step 2 matched to determine the intent). IF is_action_blocked is true, confidence_score MUST be 0.0. IF blocked, state why.\"\n"
        #     "  },\n"
        #     "  \"intent\": \"(The base_intent, ONE OF THE 4 INTENTS)\",\n"
        #     "  \"target\": \"(Extract target object name from ROBOT VISION if applicable, otherwise None)\",\n"
        #     "  \"confidence_score\": 0.9(MUST be 0.0 IF is_action_blocked is true),\n"
        #     "  \"reasoning\": \"(Explain based on the final_logic, mentioning both the gesture and the environment and the value of is_action_blocked)\"\n"
        #     "}"
        # )
        
        #cleaned up version :
        system_prompt = (
            "You are the reasoning cortex for an autonomous robot. Map the user's kinematic hand state to ONE of four intents: "
            "[PICK_UP, NAVIGATE_THERE, STOP, SEARCH_AREA].\n\n"

            "STEP 1: IDENTIFY THE HAND POSE\n"
            "Check if the hand is a known or unknown pose described in the 'HAND STATE' block.\n\n"

            "STEP 2: MAP TO INTENT (APPLY IN EXACT ORDER)\n"
            "A. HALTING (STOP):\n"
            "- IF the Hand Pose is Fist AND Spatial Motion is 'Stationary' -> Intent is STOP.\n"
            "- IF the Hand Pose is Open Palm AND Spatial Motion is 'Stationary' AND palm orientation is 'Inward' or 'Outward' -> Intent is STOP.\n\n"

            "B. SCANNING (SEARCH_AREA):\n"
            "- IF Spatial Motion contains 'Oscillating', 'Waving', or 'Rotation' -> Intent is SEARCH_AREA.\n"
            "- IF the Hand Pose is Open Palm AND Spatial Motion is 'Linear Translation' -> Intent is SEARCH_AREA.\n\n"

            "C. GRABBING (PICK_UP):\n"
            "- IF Articulation contains 'Closing', 'Grabbing', or 'Pinching' -> Intent is exclusively PICK_UP. (Overrides everything).\n"
            "- IF the Hand Pose is Pinching AND Index is touching Thumb -> Intent is PICK_UP.\n\n"

            "D. DIRECTING (NAVIGATE_THERE):\n"
            "- IF the Hand Pose is Pointing AND Spatial Motion is 'Stationary' or 'Linear Translation' -> Intent is NAVIGATE_THERE.\n"
            "- IF the Hand Pose is Open Palm AND Spatial Motion is 'Stationary' AND palm orientation is 'Down' -> Intent is NAVIGATE_THERE.\n\n"

            "STEP 3: ENVIRONMENTAL CONTEXT\n"
            "Consider the Intent matched in Step 2 and the'ROBOT VISION' context to inform the status of the action. Evaluate in this EXACT order AND skip to Step 4 as soon as a condition is met:\n\n"
            "- IF Intent is STOP, ingore the vision, the action is ALWAYS safe and possible -> action status is Safe.\n"
            "- IF Intent is SEARCH_AREA, ignore the vision, the action is ALWAYS safe and possible -> action status is Safe.\n"
            "- IF Intent is PICK_UP but the vision contains 'No objects visible', 'No box', 'Empty' or 'Obstacle'  -> The action is Blocked. \n"
            "- IF Intent is NAVIGATE_THERE but the vision contains 'Obstacle' -> The action is Blocked.\n"
            "- For all other cases, the action is safe and possible -> action status is Safe.\n\n"


            "STEP 4: OUTPUT FORMAT\n"
            "Output ONLY a valid JSON object. Do not add comments. Fill out the 'analysis' section FIRST:\n"
            "{\n"
            "  \"analysis\": {\n"
            "    \"articulation_state\": \"(Copy from log)\",\n"
            "    \"spatial_motion\": \"(Copy from log)\",\n"
            "    \"determined_pose\": \"(Write Pointing Pose, Open Palm Pose, Fist Pose, or Pinching Pose)\",\n"
            "    \"base_intent\": \"(Which Intent matched in Step 2)\",\n"
            "    \"action_status\": \"(Which action status matched in Step 3)\",\n"

            "    \"stop_override_active\": \"true or false (MUST BE true IF base_intent is STOP. Otherwise, write false)\",\n"
            "    \"is_grab_override_active\": \"true or false (MUST BE true IF articulation_state contains 'Closing' or 'Grabbing' or 'Pinching')\",\n"
            "    \"vision_context\": \"(Summarize the ROBOT VISION data)\",\n"
                        
            "    \"final_logic\": \"IF stop_override_active is true, base_intent MUST be STOP AND confidence_score MUST be 0.9. IF is_grab_override_active is true, base_intent MUST be PICK_UP (Explain which rule from Step 2 matched to determine the intent).\"\n"
            "  },\n"
            "  \"intent\": \"(The base_intent, ONE OF THE 4 INTENTS)\",\n"
            "  \"target\": \"(Extract target object name from ROBOT VISION if applicable, otherwise None)\",\n"
            # "  \"confidence_score\": 0.0,\n"
            "  \"confidence_score\":(Write 0.9 IF stop_override_active is true or action_status is 'Safe'. Write 0.0 if action_status is Blocked)\",\n"
            "  \"reasoning\": \"(Explain based on the final_logic.)\"\n"
            "}"
        )

        # --- Start Latency Timer ---
        inference_start_time = time.time()

        
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

                # self.current_confidence = prediction_json.get("confidence_score", 0.0)
                raw_confidence = prediction_json.get("confidence_score", 0.0)
                try:
                    self.current_confidence = float(raw_confidence)
                except ValueError:
                    self.current_confidence = 0.0

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
            # --- End Latency Timer ---
            inference_end_time = time.time()
            self.current_latency = inference_end_time - inference_start_time
            self.is_inferencing = False


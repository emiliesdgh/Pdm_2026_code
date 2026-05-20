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
        self.current_action_status = "Executable"

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
        # # working for search, stop & navigation however, pickup still wrong
        # # trying to force a 7-Billion parameter probabilistic language model to act as a deterministic safety logic gate.

        # system_prompt = (
        #     "You are the reasoning cortex for an autonomous robot. Map the user's kinematic hand state to ONE of four intents: "
        #     "[PICK_UP, NAVIGATE_THERE, STOP, SEARCH_AREA].\n\n"

        #     "STEP 1: IDENTIFY THE HAND POSE\n"
        #     "Check if the hand is a known or unknown pose described in the 'HAND STATE' block.\n\n"

        #     "STEP 2: MAP TO INTENT (APPLY IN EXACT ORDER)\n"
        #     "A. STOP:\n"
        #     "- IF the Hand Pose is Fist AND Spatial Motion is 'Stationary' -> Intent is STOP.\n"
        #     "- IF the Hand Pose is Open Palm AND Spatial Motion is 'Stationary' AND palm orientation is 'Inward' or 'Outward' -> Intent is STOP.\n\n"

        #     "B. SEARCH_AREA:\n"
        #     "- IF Spatial Motion contains 'Oscillating', 'Waving', or 'Rotation' -> Intent is SEARCH_AREA.\n"
        #     "- IF the Hand Pose is Open Palm AND Spatial Motion is 'Linear Translation' -> Intent is SEARCH_AREA.\n\n"

        #     "C. PICK_UP:\n"
        #     "- IF Articulation contains 'Closing', 'Grabbing', or 'Pinching' -> Intent is exclusively PICK_UP. (Overrides everything).\n"
        #     "- IF the Hand Pose is Pinching AND Index is touching Thumb -> Intent is PICK_UP.\n\n"

        #     "D. NAVIGATE_THERE:\n"
        #     "- IF the Hand Pose is Pointing AND Spatial Motion is 'Stationary' or 'Linear Translation' -> Intent is NAVIGATE_THERE.\n"
        #     "- IF the Hand Pose is Open Palm AND Spatial Motion is 'Stationary' AND palm orientation is 'Down' -> Intent is NAVIGATE_THERE.\n\n"

        #     "STEP 3: ENVIRONMENTAL CONTEXT (CRITICAL OVERRIDES)\n"
        #     "Determine if the action is Executable or Blocked based on the Intent and 'ROBOT VISION'.\n"
        #     # "- RULE 1: IF Intent is STOP or SEARCH_AREA -> Action status is ALWAYS 'Safe'. Ignore vision context completely.\n"
        #     # "- RULE 2: IF Intent is PICK_UP AND vision contains 'No objects', 'No box', 'Empty', or 'Obstacle' -> Action status is 'Blocked'.\n"
        #     "- RULE 1: IF Intent is STOP -> Action status is ALWAYS 'Executable'. (Halting is an emergency override and is highly appropriate when an obstacle is present).\n"
        #     "- RULE 2: IF Intent is SEARCH_AREA -> Action status is ALWAYS 'Executable'. (Scanning the environment is always physically possible).\n"
        #     "- RULE 3: IF Intent is NAVIGATE_THERE AND vision contains 'Obstacle' -> Action status is 'Blocked'.\n"
        #     "- RULE 4: IF Intent is PICK_UP AND vision contains 'No objects', 'No box', 'Empty', or 'Obstacle'' -> Action status is 'Blocked'.\n"
        #     "- RULE 5: For all other scenarios -> Action status is 'Executable'.\n\n"

        #     "STEP 4: OUTPUT FORMAT\n"
        #     "Output ONLY a valid JSON object. Do not add comments. Set 'confidence_score' to 0.9 if Executable, or 0.0 if Blocked. Fill out the 'analysis' section FIRST:\n"
        #     "{\n"
        #     "  \"analysis\": {\n"
        #     "    \"articulation_state\": \"(Copy from log)\",\n"
        #     "    \"spatial_motion\": \"(Copy from log)\",\n"
        #     "    \"determined_pose\": \"(Write Pointing Pose, Open Palm Pose, Fist Pose, or Pinching Pose)\",\n"
        #     "    \"base_intent\": \"(Which Intent matched in Step 2)\",\n"

        #     "    \"action_status\": \"(Write 'Executable' or 'Blocked' based on Intent and Step 3)\",\n"
        #     "    \"is_grab_override_active\": \"(true or false, MUST BE true IF articulation_state contains 'Closing' or 'Grabbing' or 'Pinching')\",\n"
        #     "    \"vision_context\": \"(Summarize the ROBOT VISION data)\",\n"

        #     # "    \"final_logic\": \"(Explain why the intent was chosen. If STOP or SEARCH_AREA, explicitly state vision was ignored. State why action is Executable or Blocked and which rule determined it.)\"\n"
        #     # "    \"final_logic\": \"(Explain why the intent was chosen. IF Intent is STOP, explicitly state that halting is Executable regardless of obstacles. State which rule determined the action status.)\"\n"
        #     "    \"final_logic\": \"IF is_grab_override_active is true, Intent MUST be PICK_UP. (Explain which rules from Step 2 and Step 3 determined the Intent.)\"\n"
        #     "  },\n"
        #     "  \"intent\": \"(The base_intent, ONE OF THE 4 INTENTS)\",\n"
        #     "  \"target\": \"(Extract target object name from vision if applicable, otherwise None)\",\n"
        #     "  \"confidence_score\": (output 0.9 if the Intent is a clear match, otherwise 0.5 if it is ambiguous),\n"
        #     "  \"reasoning\": \"(Explain based on the final_logic.)\"\n"
        #     "}"
        # )

        # shift the prompt from "Rule-Following" to "Affordance Reasoning."
        system_prompt = (
            "You are the cognitive reasoning engine for an autonomous robot. Your goal is to deduce the user's intent based on their hand kinematics and assess if the environment affords that action.\n\n"

            "--- 1. INTENT DEDUCTION (Generalization) ---\n"
            "Analyze the 'HAND STATE' and 'TEMPORAL MOTION LOG'. Do NOT rely on strict pre-defined gestures. Instead, deduce the meaning of the kinematics:\n"
            "- A hand closing, grabbing, or pinching usually implies a desire to manipulate or PICK_UP an object.\n"
            "- A pointing finger or a flat hand moving in a direction usually implies a desire to DIRECT or NAVIGATE_THERE.\n"
            "- A rigid, stationary, blocking pose (like a fist or open palm facing out) usually implies a desire to HALT or STOP.\n"
            "- Any waving, oscillating, or rotating hand, regardless of the hand pose, usually implies a desire to SCAN or SEARCH_AREA.\n"
            "Based on the kinematics, map the user's state to EXACTLY ONE intent: [PICK_UP, NAVIGATE_THERE, STOP, SEARCH_AREA].\n\n"

            "--- 2. ENVIRONMENTAL AFFORDANCE (Contextual Reasoning) ---\n"
            "Evaluate the 'ROBOT VISION' to determine if the physical environment allows the chosen intent to be executed safely.\n"
            "- Affordance of Stopping/Searching: These are internal robot states. They are ALWAYS physically possible, regardless of obstacles.\n"
            "- Affordance of Picking Up: This requires a target object. If the vision indicates 'No objects', 'Empty', or an 'Obstacle' blocking reach, the affordance is BLOCKED.\n"
            "- Affordance of Navigating: This requires a clear path. If the vision indicates 'Obstacle', 'Blocked', or 'No clear path', the affordance is BLOCKED.\n\n"

            "--- 3. OUTPUT FORMAT ---\n"
            "Output ONLY a valid JSON object. Fill out the 'analysis' section FIRST to structure your reasoning:\n"
            "{\n"
            "  \"analysis\": {\n"
            "    \"kinematic_meaning\": \"(Explain what the physical movement/pose of the hand signifies in human communication)\",\n"
            "    \"deduced_intent\": \"(State your chosen Intent based on the kinematic meaning)\",\n"
            "    \"vision_affordance\": \"(Explain if the ROBOT VISION physically supports or blocks the deduced_intent. Explicitly mention if the intent requires an object or a clear path.)\",\n"
            "    \"action_status\": \"(Write 'Executable' if the environment affords the intent, or 'Blocked' if it does not)\"\n"
            "  },\n"
            "  \"intent\": \"(MUST be exactly PICK_UP, NAVIGATE_THERE, STOP, or SEARCH_AREA)\",\n"
            "  \"target\": \"(Extract the object name if applicable, otherwise None)\",\n"
            "  \"confidence_score\": (0.9 if the kinematics strongly match the intent AND it is Executable. 0.0 if the action is Blocked. 0.5 if the intent is ambiguous.),\n"
            "  \"reasoning\": \"(Summarize the kinematic meaning and the vision affordance.)\"\n"
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
                self.current_action_status = prediction_json.get("analysis", {}).get("action_status", "Executable")

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


import ollama
import threading
import json

class LLMInferenceAgent:
    # Mixtral requires 24GB to 32GB RAM
    # if the computer doesn't have enough RAM, it might crash
    # ==> in this case, use Mistral
    # before using, run in terminal: ollama pull mixtral
    # --> this allows to download the model
    # def __init__(self, model_name="mixtral"):

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

    
        # if self.is_inferencing:
        #     print("Already inferencing, please wait...")
        #     return
        
        # def inference_thread():
        #     self.is_inferencing = True
        #     try:
        #         response = ollama.chat(self.model_name, symbolic_str)
        #         self.current_prediction = response
        #         print(f"LLM Prediction: {response}")
        #     except Exception as e:
        #         print(f"Error during LLM inference: {e}")
        #         self.current_prediction = "Error during inference."
        #     finally:
        #         self.is_inferencing = False

        # threading.Thread(target=inference_thread).start()

    def _query_ollama(self, symbolic_str):
        """
        Function that send the prompt to ollama and get the response, runs in a separate thread.
        the system prompt is defined here to avoid re-defining it every time in the main loop, and to have a single source of truth for the system prompt.
        it gives the context to the LLM about the task and what it should do with the symbolic string that is sent from the main loop.
        """
        # system_prompt = """
        # You are the cognitive Intent Inference Layer for a humanoid robot. 
        # Your task is to interpret human body language and map it to the robot's available actions.

        # ROBOT CAPABILITIES (The Intent Space):
        # 1. "PICK_UP": The robot grasps or manipulates an object.
        # 2. "NAVIGATE_THERE": The robot walks to a specific location or direction.
        # 3. "SEARCH_AREA": The robot looks around to find something.
        # 4. "STOP": The robot immediately halts all action.

        # INSTRUCTIONS:
        # Read the kinematic state of the human's hand. Do not rely on strict rules. Instead, use common sense human intuition to infer what the human wants the robot to do. 
        # - A deictic gesture (pointing, indicating a direction) usually implies navigation or drawing attention.
        # - An iconic gesture (mimicking an action like grabbing, pushing, or waving) usually implies manipulation or searching.
        # - A flat, outward-facing palm is a universal human sign for halting or rejection.

        # Respond ONLY with a JSON object: {"intent": string, "reasoning": string}
        # """
                                                                # reasoning to debeug why the LLM made such predictions, to notice the understanding of the LLM
        
        # system_prompt = """
        # You are the cognitive Intent Inference Layer for a humanoid robot. 
        # Your task is to interpret the kinematic state of a human's hand and map it to the robot's available actions.

        # ROBOT CAPABILITIES (The Intent Space):
        # 1. "PICK_UP": The robot grasps or manipulates an object.
        # 2. "NAVIGATE_THERE": The robot walks to a specific location or direction.
        # 3. "SEARCH_AREA": The robot looks around to find something.
        # 4. "STOP": The robot immediately halts all action.

        # KINEMATIC TRANSLATION GUIDE:
        # The hand tracking system uses technical terms. Use this guide to understand them:
        # - "Oscillating Left & Right" or "Hand Rotation" -> Usually represents a Waving or Sweeping motion.
        # - "Linear Translation" -> The hand is moving in a line (Swiping or Reaching).
        # - "Bending Fingers" -> The hand is actively opening or closing.
        # - "Stationary" (or "Slow" movements) -> The hand is mostly holding a pose.

        # REASONING INSTRUCTIONS:
        # Use common sense to infer intent based on the finger state and motion.
        # - A pointing index finger, even with slight "Linear Translation", implies indicating a direction (NAVIGATE_THERE).
        # - An open, flat palm (all fingers straight), especially if mostly stationary, is a universal sign for halting (STOP).
        # - A waving motion ("Oscillating" or "Hand Rotation"), even if some fingers are bent, usually implies wanting the robot to look around (SEARCH_AREA).
        # - A fist (fingers bent) or a "Bending Fingers" motion implies grasping or manipulating an object (PICK_UP).

        # Respond ONLY with a JSON object: {"intent": string, "reasoning": string}
        # """

        # system_prompt = """
        # You are the cognitive Intent Inference Layer for a humanoid robot. 
        # Your task is to interpret the kinematic state of a human's hand and map it to the robot's available actions.

        # ROBOT CAPABILITIES (The Intent Space):
        # 1. "PICK_UP": The robot grasps, lifts, or manipulates an object.
        # 2. "NAVIGATE_THERE": The robot walks to a specific location or direction.
        # 3. "SEARCH_AREA": The robot looks around or scans the environment.
        # 4. "STOP": The robot immediately halts all action.

        # KINEMATIC TRANSLATION GUIDE:
        # - "Oscillating Left & Right" or "Hand Rotation" -> Waving, sweeping, scanning, or drawing a circle.
        # - "Linear Translation" -> Reaching, swiping, or pulling in a specific direction.
        # - "Bending Fingers" -> The active motion of grabbing, closing the hand, or squeezing.
        # - "Stationary" -> Holding a rigid pose or commanding a halt.

        # REASONING INSTRUCTIONS (Accommodating Human Variation):
        # Humans express intents differently. Use these principles to guide your reasoning:
        # - NAVIGATE_THERE: Usually an extended index finger (pointing). It is typically stationary, but might involve a slight linear translation to indicate a path.
        # - STOP: A defensive command to halt. It is strictly "Stationary". While often a flat, outward-facing palm, humans also use a rigid, stationary closed fist held in the air to mean "Hold" or "Stop".
        # - SEARCH_AREA: Implies scanning. The key indicator is the motion ("Oscillating" or "Hand Rotation"). The hand might be open, OR the user might use a pointing index finger while rotating the hand to "draw" a circle in the air. 
        # - PICK_UP: Implies grabbing or lifting. It strongly relies on dynamic motion. Look for "Bending Fingers" (closing the hand) OR an upward "Linear Translation" combined with folded fingers (mimicking lifting an object).

        # Respond ONLY with a JSON object: {"intent": string, "reasoning": string}
        # """
        # system_prompt = (
        #     "You are the visual reasoning cortex for an autonomous robot. Your task is to map the user's hand state to ONE of four specific robot skills and understanding the user's intent: "
        #     "[PICK_UP, NAVIGATE_THERE, STOP, SEARCH_AREA].\n\n"

        #     "These are some guidelines to decode the human's intent based on their hand state:\n"

        #     "1. NAVIGATE_THERE (Deictic): Most ofthen involves pointing (i.e., Index finger extended, others folded, the thumb might be extended as well, but having middle, ring and pinky folded in contrast to the index extended is the important thing to note).The index must be extended for pointing otherwise it is not pointing. The hand is more often stationary or swiping towards a direction.\n"

        #     "2. SEARCH_AREA (Iconic/Deictic): The user wants the robot to look around. They might point and rotate their hand (scanning), or have all fingers mostly extended and making a swiping motion in the scene, or make oscillating/waving motions. They most likely will have their hand facing downwards and they most likely won't have all fingers bent in a fist.\n"
            
        #     "3. PICK_UP (Iconic): The user is mimicking grabbing. Look for 'Bending Fingers', 'Hand Open/Close', or all finger tips touching the thumb tip, often paired with upward motions or palm facing down.\n"
            
        #     "4. STOP (Iconic): Usually a static/ stationnary gesture to halt the robot. This could be a static Fist (all fingers folded or fingers folded with an extended thumb) or an Open Palm facing mostly outward (all fingers extended) with no motion. If the hand has a motion other than stationary, it it probably NOT a stop gesutre.\n\n"
            
        #     "Do not assume 'bent fingers' always means grabbing, especially if the Index is extended (which implies pointing). "

        #     "If all fingers are bent, none can be extended. If only the Index (and possibly the Thumb), but the other fingers (Middle, Ring and Pinky) are bent, the user is most likely pointing. If the Index is NOT extended, the user is most likely NOT pointing."

        #     "Output a JSON with two keys: 'intent' (the exact name of the command) and 'reasoning'."
        # )

        system_prompt = (
            "You are the visual reasoning cortex for an autonomous robot. Your task is to map the user's hand state to ONE of four specific robot skills and understanding the user's intent: " \
            "[PICK_UP, NAVIGATE_THERE, STOP, SEARCH_AREA].\n\n" 
            
            "These are some guidelines to decode the human's intent based on their hand state:\n" 

            "1. NAVIGATE_THERE (Diectic): Stationary mostion. Most often involves pointing (extended Index, Middle, Ring and Pinky bent). The hand is more often stationary. The Thumb state (bent or extended) might vary with the user. \n"
            
            "2. SEARCH_AREA (Iconic/Deictic): The user wants the robot to look around. They might perform a sweeping motion with an open hand (all fingers extended) or a pointing hand (Index extended) while rotating the hand back and forth or around, making wrist rotations in the plane (oscillating motion). They most likely will have their hand facing downwards and they most likely won't have all fingers bent in a fist because they might not make a sweeping motion in a fist.\n"

            "3. PICK_UP (Iconic): The user is mimicking grabbing. Look for 'Bending Fingers', 'Hand Open/Close', or all finger tips touching the thumb tip, often paired with translation motions (usually up/down but can be slightly diagonal) or palm facing down. Look for the pinching motion of bending extended fingers or bringing all finger tips to the Thumb.\n"

            "4. STOP (Iconic): Usually a stationary gesture to halt the robot. Some users might do a static Fist (all fingers folded or fingers folded with an extended thumb) while others might do an Open Palm facing mostly outward (all fingers extended) with no motion. If the hand has a motion other than stationary, it is probably NOT a stop gesture. The stationary is an important indicator.\n\n"

            "Here are some other indicators to help you decode the intent:\n"
            
            "- If, out of the four fingers (Index, Middle, Ring, Pinky), only the Index is extended while the others are bent, the user is most likely pointing. The Thumb might be bent or extended depending on the user. If the Index is NOT extended, the user is most likely NOT pointing. The extended Index is a key indicator.\n"
             
            "- If all finger are bent, none can be extended.\n"
            
            "- Do not assume 'bent fingers' always means grabbing, especially if the Index is extended (which most probablyimplies pointing). \n\n"

            

            "Output a JSON with two keys: 'intent' (the exact name of the command) and 'reasoning'."
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
                return prediction_json
            
            except json.JSONDecodeError:
                print(f"❌ JSON Parse Error. Raw Output: {response_text}")
                # return {"intent": "UNKNOWN"}
                self.current_intent = "UNKNOWN"
                return {"intent": "UNKNOWN", "reasoning": "Failed to parse LLM response."}

        except Exception as e:
            print(f"❌ Ollama Error: {e}")
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


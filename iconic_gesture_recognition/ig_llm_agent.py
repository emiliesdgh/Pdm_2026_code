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
        self.current_prediction = "Waiting for hand gesture..."

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
        
        system_prompt = """
        You are the cognitive Intent Inference Layer for a humanoid robot. 
        Your task is to interpret the kinematic state of a human's hand and map it to the robot's available actions.

        ROBOT CAPABILITIES (The Intent Space):
        1. "PICK_UP": The robot grasps or manipulates an object.
        2. "NAVIGATE_THERE": The robot walks to a specific location or direction.
        3. "SEARCH_AREA": The robot looks around to find something.
        4. "STOP": The robot immediately halts all action.

        KINEMATIC TRANSLATION GUIDE:
        The hand tracking system uses technical terms. Use this guide to understand them:
        - "Oscillating Left & Right" or "Hand Rotation" -> Usually represents a Waving or Sweeping motion.
        - "Linear Translation" -> The hand is moving in a line (Swiping or Reaching).
        - "Bending Fingers" -> The hand is actively opening or closing.
        - "Stationary" (or "Slow" movements) -> The hand is mostly holding a pose.

        REASONING INSTRUCTIONS:
        Use common sense to infer intent based on the finger state and motion.
        - A pointing index finger, even with slight "Linear Translation", implies indicating a direction (NAVIGATE_THERE).
        - An open, flat palm (all fingers straight), especially if mostly stationary, is a universal sign for halting (STOP).
        - A waving motion ("Oscillating" or "Hand Rotation") usually implies wanting the robot to look around (SEARCH_AREA).
        - A fist (fingers bent) or a "Bending Fingers" motion implies grasping or manipulating an object (PICK_UP).

        Respond ONLY with a JSON object: {"intent": string, "reasoning": string}
        """
        
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
                return prediction_json
            except json.JSONDecodeError:
                print(f"❌ JSON Parse Error. Raw Output: {response_text}")
                return {"intent": "UNKNOWN"}

        except Exception as e:
            print(f"❌ Ollama Error: {e}")
            return {"intent": "UNKNOWN"}
        
        #     self.current_prediction = response['message']['content'].strip()
        #     print(f"\n[LLM Prediction]: {self.current_prediction}\n")
        # except Exception as e:
        #     print(f"Ollama Error: {e}")
        # finally:
        #     self.is_inferencing = False


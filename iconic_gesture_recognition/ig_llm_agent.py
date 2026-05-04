import ollama
import threading

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
        # system_prompt = (
        #     "You are a helpful assistant for recognizing hand gestures based on the hand state information provided. "
        #     "The user will perform a hand gesture, and you will receive a detailed description of the hand's state, including finger positions, contacts, orientation, and motion. "
        #     "Based on this information, your task is to infer the most likely gesture being performed by the user. "
        #     "You should consider common gestures such as 'Thumbs Up', 'Peace Sign', 'Fist', 'Open Hand', 'Pointing', etc., but also be open to less common gestures based on the provided hand state. "
        #     "Your response should be concise and focused on identifying the gesture."
        # )

        # 1. Aggressive System Prompt
        # system_prompt = (
        #     "You are a gesture recognition classifier. Read the hand state and output ONLY the name of the gesture. "
        #     "Do NOT write full sentences. Do NOT explain your reasoning. "
        #     "Example outputs: 'Thumbs Up', 'Waving', 'Fist', 'Open Hand', 'Pointing', 'Swipe Left', 'Unknown'."
        # )

        system_prompt = (
            "System: You are an HRI Intent Interpreter for a humanoid robot."
            "Context: The robot is in a lab. There are boxes on a table."
            f"User Input: {symbolic_str}"

            "Task: Infer the user's intent. Here are some examples of intents you might infer based on the user's hand gesture: "
            " - If they mimic a 'grasp' (all fingers closing), intent is \"PICK_UP\"."
            " - If they point, intent is \"NAVIGATE\" to that coordinate."
            " - If they move their hand in a circle, intent is \"SEARCH_AREA\"."

            "Return JSON: {\"intent\": string, \"spatial_hint\": [x, y] or null, \"confidence\": 0.0-1.0}"
        )

        try:
            response =ollama.chat(model=self.model_name, messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': symbolic_str}
            ], options = {'num_predict': 20})

            self.current_prediction = response['message']['content'].strip()
            print(f"\n[LLM Prediction]: {self.current_prediction}\n")
        except Exception as e:
            print(f"Ollama Error: {e}")
        finally:
            self.is_inferencing = False


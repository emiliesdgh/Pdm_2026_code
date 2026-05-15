import json
from ig_llm_agent import LLMInferenceAgent
from ig_logger import setup_logger
from pathlib import Path

# dataset_path = Path("datasets_and_logs/gesture_dataset.json")
dataset_path = Path("gesture_dataset_v2.json")

# logger = setup_logger("gesture_log_evaluator.txt")

def evaluate_llm():
    agent = LLMInferenceAgent(model_name="mistral")
    # agent = LLMInferenceAgent(model_name="mixtral")   # too slow and not necessarily better
    
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    correct = 0
    total = len(dataset)

    for data in dataset:
        print(f"\nTesting ID: {data['id']}")
        print(f"Input: {data['symbolic_string']}")

        logger.info(f"Testing ID: {data['id']}")
        logger.info(f"Input: {data['symbolic_string']}")
        
        # We use a synchronous call here for testing so it waits for the answer
        # You will need to modify _query_ollama to return the response for this test
        prediction_json = agent._query_ollama(data['symbolic_string']) 
        
        predicted_intent = prediction_json.get("intent", "UNKNOWN")
        ground_truth = data['ground_truth']
        reasoning = prediction_json.get("reasoning", "No reasoning provided")
        print(f"LLM reasoning: {reasoning}")
        logger.info(f"LLM reasoning: {reasoning}")

        if predicted_intent == ground_truth:
            print("✅ CORRECT")
            logger.info("✅ CORRECT")
            correct += 1
        else:
            print(f"❌ FAILED. Expected: {ground_truth}, Got: {predicted_intent}")
            logger.info(f"❌ FAILED. Expected: {ground_truth}, Got: {predicted_intent}")

    accuracy = (correct / total) * 100
    print(f"\n--- FINAL ACCURACY: {accuracy}% ---")
    logger.info(f"\n--- FINAL ACCURACY: {accuracy}% ---")
    

if __name__ == "__main__":
    logger = setup_logger("gesture_log_evaluator_live_test3.txt")

    evaluate_llm()
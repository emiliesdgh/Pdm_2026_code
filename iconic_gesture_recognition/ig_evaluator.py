import json
from ig_llm_agent import LLMInferenceAgent

def evaluate_llm():
    agent = LLMInferenceAgent(model_name="mistral")
    
    with open("gesture_dataset.json", "r") as f:
        dataset = json.load(f)

    correct = 0
    total = len(dataset)

    for data in dataset:
        print(f"\nTesting ID: {data['id']}")
        print(f"Input: {data['symbolic_string']}")
        
        # We use a synchronous call here for testing so it waits for the answer
        # You will need to modify _query_ollama to return the response for this test
        prediction_json = agent._query_ollama(data['symbolic_string']) 
        
        predicted_intent = prediction_json.get("intent", "UNKNOWN")
        reasoning = prediction_json.get("reasoning", "No reasoning provided.")
        ground_truth = data['ground_truth']
        print(f"Reasoning: {reasoning}")
        
        if predicted_intent == ground_truth:
            print("✅ CORRECT")
            correct += 1
        
        else:
            print(f"❌ FAILED. Expected: {ground_truth}, Got: {predicted_intent}")

    accuracy = (correct / total) * 100
    print(f"\n--- FINAL ACCURACY: {accuracy}% ---")

if __name__ == "__main__":
    evaluate_llm()
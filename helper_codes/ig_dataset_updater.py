import json
from pathlib import Path


def update_dataset_strings(input_file="gesture_dataset_good.json", output_file="gesture_dataset_v2.json"):
    with open(input_file, "r") as f:
        dataset = json.load(f)

    updated_count = 0

    for data in dataset:
        original_string = data["symbolic_string"]
        lines = original_string.split('\n')
        
        is_thumb_straight = False
        contact_line_index = -1
        
        # 1. Analyze the lines to find the flexion and contact states
        for i, line in enumerate(lines):
            # Check if the Thumb is straight
            if "All fingers are straight" in line:
                is_thumb_straight = True
            elif "straight" in line and "bent" in line:
                # Look at the first half of the sentence (the straight fingers)
                straight_part = line.split("straight")[0]
                if "Thumb" in straight_part:
                    is_thumb_straight = True
            
            # Find where the contact line is
            if "The Thumb is currently in contact with" in line:
                contact_line_index = i
                
        # 2. Apply your new ig_inference.py rule: 
        # If Thumb is straight, it CANNOT be in contact
        if is_thumb_straight and contact_line_index != -1:
            # Overwrite the contact line with the exact string from your updated script
            lines[contact_line_index] = "- The Thumb is NOT in contact with fingertips."
            updated_count += 1
            
        # Rejoin the string and update the dataset in memory
        data["symbolic_string"] = '\n'.join(lines)

    # 3. Save to a new file so you don't lose your original data
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=4)

    print(f"Dataset updated successfully! Fixed {updated_count} contradicting gestures.")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    update_dataset_strings()
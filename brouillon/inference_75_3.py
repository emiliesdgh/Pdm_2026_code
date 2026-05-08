"""
Script to link the gesture recognition to the LLM symbolic representation by determining the hand state (finger positions, hand orientation, etc.)
"""





def get_symbolic_string_2(global_vars, finger_flexion_state, finger_contact_state, hand_orientation, motion_detected, motion_type, hand_position):
    """
    Formats the hand state into descriptive bullet points modeled after the GestureGPT paper.
    """
    # 1. Group fingers by state
    extended = [name.capitalize() for name, val in zip(global_vars.FINGERS["name"], finger_flexion_state) if val == 1]
    folded = [name.capitalize() for name, val in zip(global_vars.FINGERS["name"], finger_flexion_state) if val == -1]
    
    if extended and folded:
        flexion_desc = f"The {', '.join(extended)} fingers are straight, while the {', '.join(folded)} fingers are bent."
    elif extended:
        flexion_desc = "All fingers are straight."
    else:
        flexion_desc = "All fingers are bent."

    # 2. Group finger contacts
    in_contact = [name.capitalize() for name, val in zip(global_vars.FINGERS["name"][1:], finger_contact_state) if val == 1]
    if in_contact:
        contact_desc = f"The Thumb is currently in contact with the {', '.join(in_contact)} fingertips."
    else:
        contact_desc = "The Thumb is not in contact with any other fingertips."

    # 3. Format the final bulleted prompt
    symbolic_str = (
        "Here is the current state of the user's hand:\n"
        f"- {flexion_desc}\n"
        f"- {contact_desc}\n"
        f"- The palm orientation is facing {hand_orientation}.\n"
        f"- The hand is positioned at {hand_position} relative to the center of the view.\n"
        f"- The hand moves with a {motion_type} motion."
    )

    # print(symbolic_str)

    return symbolic_str
"""
Script to link the gesture recognition to the LLM symbolic representation by determining the hand state (finger positions, hand orientation, etc.)
"""

def get_symbolic_string_2(global_vars, finger_flexion_state, finger_contact_state, hand_orientation, motion_detected, spatial_motion, articulation, hand_position, environmental_context=""):
    """
    Formats the hand state into descriptive bullet points modeled after the GestureGPT paper.
    """
    # 1. Group fingers by state
    extended = [name.capitalize() for name, val in zip(global_vars.FINGERS["name"], finger_flexion_state) if val == 1]
    folded = [name.capitalize() for name, val in zip(global_vars.FINGERS["name"], finger_flexion_state) if val == -1]
    
    if extended and folded:
        if len(extended)>=2 and len(folded)==1:
            flexion_desc = f"The {', '.join(extended)} fingers are straight, while the {', '.join(folded)} finger is bent."
        elif len(extended)==1 and len(folded)>=2:
            flexion_desc = f"The {', '.join(extended)} finger is straight, while the {', '.join(folded)} fingers are bent."
        elif len(extended)>=2 and len(folded)>=2:
            flexion_desc = f"The {', '.join(extended)} fingers are straight, while the {', '.join(folded)} fingers are bent."
        else:
            flexion_desc = f"The {', '.join(extended)} finger is straight, while the {', '.join(folded)} finger is bent."
        # check if do a condition for several fingers for the conjugaison
    elif extended:
        flexion_desc = "All fingers are straight."
    else:
        flexion_desc = "All fingers are bent."

    # 2. Group finger contacts
    in_contact = [name.capitalize() for name, val in zip(global_vars.FINGERS["name"][1:], finger_contact_state) if val == 1]
    if in_contact:
        if len(in_contact) == 1:
            contact_desc = f"The Thumb is currently in contact with the {', '.join(in_contact)} fingertip."
        else:
            contact_desc = f"The Thumb is currently in contact with the {', '.join(in_contact)} fingertips."
    else:
        contact_desc = "The Thumb is NOT in contact with fingertips."#any other fingertips."

    if spatial_motion == "Stationary":
        motion_desc = "The hand is stationary, motionless."
    else:
        motion_desc = f"The hand is moving with a {spatial_motion} motion."

    # 3. Format the final bulleted prompt
    symbolic_str = (
        "Here is the current state of the user's hand:\n"
        f"- {flexion_desc}\n"
        f"- {contact_desc}\n"
        f"- The palm orientation is facing {hand_orientation}.\n"
        f"- The hand is positioned at {hand_position} relative to the center of the view.\n"
        f"--- TEMPORAL MOTION LOG ---\n"
        # f"- Spatial Motion: {spatial_motion}\n"
        f"- Spatial Motion: {motion_desc}\n"
        f"- Articulation: {articulation}\n\n"
        f"--- ROBOT VISION (ENVIRONMENTAL CONTEXT) ---\n"
        f"{environmental_context if environmental_context else 'No additional context from robot vision.'}"
    )

    # print(symbolic_str)

    return symbolic_str
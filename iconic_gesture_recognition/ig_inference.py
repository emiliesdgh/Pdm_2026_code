"""
Script to link the gesture recognition to the LLM symbolic representation by determining the hand state (finger positions, hand orientation, etc.)
"""

def get_symbolic_string_2(global_vars, finger_flexion_state, finger_contact_state, hand_orientation, motion_detected, spatial_motion, articulation, hand_position, sensor_confidence, environmental_context=""):
    """
    Formats the hand state into descriptive bullet points modeled after the GestureGPT paper.
    """
    ### === Finger Flexion State === ###
    extended = [name.capitalize() for name, val in zip(global_vars.FINGERS["name"], finger_flexion_state) if val == 1]
    folded = [name.capitalize() for name, val in zip(global_vars.FINGERS["name"], finger_flexion_state) if val == -1]
    hand_pose = "Unknown"
    
    if extended and folded:
        if len(extended)>=2 and len(folded)==1:
            flexion_desc = f"The {', '.join(extended)} fingers are straight, while the {', '.join(folded)} finger is bent."
        elif len(extended)==1 and len(folded)>=2:
            flexion_desc = f"The {', '.join(extended)} finger is straight, while the {', '.join(folded)} fingers are bent."

        elif len(extended)>=2 and len(folded)>=2:
            flexion_desc = f"The {', '.join(extended)} fingers are straight, while the {', '.join(folded)} fingers are bent."
            if len(extended) == 2 and "Thumb" in extended and "Index" in extended:
                hand_pose = "Pointing"
        else:
            flexion_desc = f"The {', '.join(extended)} finger is straight, while the {', '.join(folded)} finger is bent."
        # check if do a condition for several fingers for the conjugaison
    elif extended:
        if len(extended) == 5:
            flexion_desc = "All fingers are straight."
            hand_pose = "Open Palm"
        else: 
            flexion_desc = f"The {', '.join(extended)} fingers are straight, while the other fingers are relaxed."
    elif folded:
        if len(folded) == 5:
            flexion_desc = "All fingers are bent."
            hand_pose = "Fist"
        else:
            flexion_desc = f"The {', '.join(folded)} fingers are bent, while the other fingers are relaxed."
    # should we interprete relaxed as straight ? so that only when the fingers are bent they are bent and otherwise taken into account as if straight for the intent prediction ?
    else:
        flexion_desc = "All fingers are in a relaxed, natural state."

    

    ### === Contact State (Thumb) === ###
    in_contact = [name.capitalize() for name, val in zip(global_vars.FINGERS["name"][1:], finger_contact_state) if val == 1]
    # if in_contact and flexion_desc != "All fingers are straight.":  # Only mention contact if not all fingers are straight (since that would be contradictory)
    if in_contact and finger_flexion_state[0] == -1:  # Only mention contact if thumb is bent (otherwise i can get contradictory)
        if len(in_contact) == 1:
            contact_desc = f"The Thumb is currently in contact with the {', '.join(in_contact)} fingertip."
        else:
            contact_desc = f"The Thumb is currently in contact with the {', '.join(in_contact)} fingertips."
    else:
        # the thumb cannot have a contact if all fingers are straight
        contact_desc = "The Thumb is NOT in contact with fingertips."#any other fingertips."

    ### === Spatial Motion === ###
    if spatial_motion == "Stationary":
        motion_desc = "The hand is stationary, motionless."
    else:
        motion_desc = f"The hand is moving with a {spatial_motion} motion."

    ### === Hand Pose === ###
    if len(extended) == 1 and "Index" in extended:
        hand_pose = "Pointing"
    elif len(extended)==2 and "Thumb" in extended and "Index" in extended:
        hand_pose = "Pointing"

    if len(in_contact) != 0 and "Index" in in_contact:
        hand_pose = "Pinching"

    if hand_pose != "Unknown":
        hand_state = f"The hand is in a {hand_pose} pose."
    else:
        hand_state = f"{flexion_desc}"

    # print(f"DEBUG: hand_pose={hand_pose}, flexion_desc={flexion_desc}")

    # 3. Format the final bulleted prompt
    symbolic_str = (
        f"--- SENSOR CONFIDENCE ---\n"
        f"Camera Tracking Confidence: {sensor_confidence:.2f}/1.0\n\n"
        
        f"--- TEMPORAL MOTION LOG ---\n"
        # f"- Spatial Motion: {spatial_motion}\n"
        f"- Spatial Motion: {motion_desc}\n"
        f"- Articulation: {articulation}\n\n"
        # trying to see if having articulation at the top makes it more salient for the LLM because sometimes
        # it hallucinates when it is grabing or pinching as something else like stop

        # "Here is the current state of the user's hand:\n"
        "--- HAND STATE ---\n"
        f"- {hand_state}\n"
        f"- {contact_desc}\n"
        f"- The palm orientation is {hand_orientation}.\n"
        f"- The hand is positioned at {hand_position} relative to the center of the view.\n\n"
        
        f"--- ROBOT VISION (ENVIRONMENTAL CONTEXT) ---\n"
        f"{environmental_context if environmental_context else 'No additional context from robot vision.'}"
    )

    # print(symbolic_str)

    return symbolic_str




"""
--- SENSOR CONFIDENCE ---
Camera Tracking Confidence: 1.00/1.0

--- TEMPORAL MOTION LOG ---
- Spatial Motion: The hand is stationary, motionless.
- Articulation: Closing

--- HAND STATE ---
- The hand is in a Fist pose.
- The Thumb is NOT in contact with fingertips.
- The palm orientation is Inwards.
- The hand is positioned at [x, y, z] relative to the center of the view.

--- ROBOT VISION (ENVIRONMENTAL CONTEXT) ---
No additional context from robot vision.

"""
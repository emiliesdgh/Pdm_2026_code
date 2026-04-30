"""
Script to link the gesture recognition to the LLM symbolic representation by determining the hand state (finger positions, hand orientation, etc.)
"""


def get_symbolic_string(global_vars, finger_flexion_state, finger_contact_state, hand_orientation, motion_detected, motion_type, hand_position):
    """
    Converts raw detection data into a natural language description.
    """    
    # Map 1/0 to EXTENDED/FOLDED for better LLM reasoning
    mapping_ext_fold = {1: 'EXTENDED', -1: 'FOLDED', 0: 'UNSURE'}
    states = [f"{name}: {mapping_ext_fold.get(val, 'UNKNOWN')}" for name, val in zip(global_vars.FINGERS["name"], finger_flexion_state)]

    finger_description = ", ".join(states)

    mapping_contact = {1: 'YES', -1: 'NO', 0: 'UNSURE'}
    contact_desc = [f"{name}: {mapping_contact.get(val, 'UNKNOWN')}" for name, val in zip(global_vars.FINGERS["name"][1:], finger_contact_state)]
    contact_results = ", ".join(contact_desc)
    
    # motion_type comes from TemporalGestureManager
    symbolic_str = (
        f"Fingers Description: {finger_description} \n"
        f"Finger in Contact with Thumb: {contact_results} \n" 
        f"Hand Orientation: {hand_orientation} \n" 
        f"Hand Position: {hand_position} \n"    # Add hand position information, how to have it not in coordinates ?
        f"Is a Motion Detected: {motion_detected} \n"
        f"Motion Type: {motion_type} \n"
        
    )
    print(symbolic_str)
    
    return symbolic_str

def get_symbolic_string_2(global_vars, finger_flexion_state, finger_contact_state, hand_orientation, motion_detected, motion_type, hand_position):
    """
    Alternative version to have more of a descriptive sentences rather than a list of attributes, might be better for the LLM to reason about the gesture
    """
    # Map 1/0/-1 to Extended / Folded / Unsure for better LLM reasoning
    mapping_ext_fold = {1: 'extended', -1: 'folded', 0: 'unsure'}
    extended = [name for name, val in zip(global_vars.FINGERS["name"], finger_flexion_state) if val == 1]
    folded = [name for name, val in zip(global_vars.FINGERS["name"], finger_flexion_state) if val == -1]
    unsure_fold = [name for name, val in zip(global_vars.FINGERS["name"], finger_flexion_state) if val == 0]


    finger_description = ", ".join(extended) + " are extended." if extended else "All fingers folded."

    mapping_contact = {1: 'YES', -1: 'NO', 0: 'UNSURE'}
    in_contact = [name for name, val in zip(global_vars.FINGERS["name"][1:], finger_contact_state) if val == 1]
    contact_results = ", ".join(in_contact) + " are in contact with the thumb." if in_contact else "No fingers in contact with the thumb."



    symbolic_str = (
        f"Finger States: \n"
        f"{finger_description}  \n"
        f"{contact_results} \n"
        f"\n"
        f"Hand Orientation: The hand is facing the {hand_orientation} direction. \n"
        # f"Hand Position: {hand_position} \n"    # Add hand position information
        f"\n"
        f"Motion Detected: {'Yes' if motion_detected else 'No'}, the hand is moving. \n"
        f"Motion Type: The motion that the hand is performing is a {motion_type} \n"
    )

    print(symbolic_str)

    return symbolic_str
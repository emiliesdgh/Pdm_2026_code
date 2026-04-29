"""
Script to link the gesture recognition to the LLM symbolic representation by determining the hand state (finger positions, hand orientation, etc.)
"""

FINGERS ={
    "name": ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"],
    "tip_idx": [4, 8, 12, 16, 20],
    "dip_idx": [3, 7, 11, 15, 19],
    "pip_idx": [2, 6, 10, 14, 18],
    "base_idx": [1, 5, 9, 13, 17]
}

def get_symbolic_string(finger_flexion_state, finger_contact_state, hand_orientation, motion_detected, motion_type, hand_position):
    """
    Converts raw detection data into a natural language description.
    """
    # finger_names = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]
    
    # Map 1/0 to EXTENDED/FOLDED for better LLM reasoning
    mapping_ext_fold = {1: 'EXTENDED', -1: 'FOLDED', 0: 'UNSURE'}
    states = [f"{name}: {mapping_ext_fold.get(val, 'UNKNOWN')}" for name, val in zip(FINGERS["name"], finger_flexion_state)]

    finger_description = ", ".join(states)

    mapping_contact = {1: 'YES', -1: 'NO', 0: 'UNSURE'}
    contact_desc = [f"{name}: {mapping_contact.get(val, 'UNKNOWN')}" for name, val in zip(FINGERS["name"][1:], finger_contact_state)]
    contact_results = ", ".join(contact_desc)
    
    # motion_type comes from TemporalGestureManager
    symbolic_str = (
        # f"Hand Description: {finger_description}. \n"
        # f"Finger in Contact with Thumb: {contact_results}. \n" # would need it to be yes, no or maybe rather than 1, 0, -1
        # f"Hand Orientation Angle: {hand_orientation}. \n" # would need it in degrees ?
        # f"Is a Motion Detected: {motion_detected}. \n"
        # f"Motion Type: {motion_type}. \n"
        # f"Hand Position: {hand_position}."
    )
    print(symbolic_str)
    
    return symbolic_str
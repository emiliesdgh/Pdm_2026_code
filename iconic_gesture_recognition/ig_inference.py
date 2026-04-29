"""
Script to link the gesture recognition to the LLM symbolic representation by determining the hand state (finger positions, hand orientation, etc.)
"""

def get_symbolic_string(flexion_results, contact_results, palm_orientation, motion_detected, motion_type, hand_position):
    """
    Converts raw detection data into a natural language description.
    """
    # finger_names = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]
    
    # Map 1/0 to EXTENDED/FOLDED for better LLM reasoning
    mapping_ext_fold = {1: 'EXTENDED', -1: 'FOLDED', 0: 'UNSURE'}
    states = [f"{name}: {mapping_ext_fold.get(val, 'UNKNOWN')}" for name, val in flexion_results.items()]
    
    finger_description = ", ".join(states)

    mapping_contact = {1: 'YES', -1: 'NO', 0: 'UNSURE'}
    contact_desc = [f"{name}: {mapping_contact.get(val, 'UNKNOWN')}" for name, val in contact_results.items()]
    contact_results = ", ".join(contact_desc)
    
    # motion_type comes from TemporalGestureManager
    symbolic_str = (
        f"Hand Description: {finger_description}. \n"
        f"Finger Contact (with Thumb): {contact_results}. \n" # would need it to be yes, no or maybe rather than 1, 0, -1
        f"Palm Orientation Angle: {palm_orientation}. \n" # would need it in degrees ?
        f"Is a Motion Detected: {motion_detected}. \n"
        f"Motion Type: {motion_type}. \n"
        f"Hand Position: {hand_position}."
    )
    print(symbolic_str)
    
    return symbolic_str
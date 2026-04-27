"""
Script to link the gesture recognition to the LLM symbolic representation by determining the hand state (finger positions, hand orientation, etc.)
"""

def generate_symbolic_representation(fingers_state, orientation, motion_type):
    """
    Converts raw detection data into a natural language description.
    """
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    
    # Map 1/0 to EXTENDED/FOLDED for better LLM reasoning
    states = [
        f"{finger_names[i]}: {'EXTENDED' if s == 1 else 'FOLDED'}" 
        for i, s in enumerate(fingers_state)
    ]
    
    finger_description = ", ".join(states)
    
    # motion_type comes from TemporalGestureManager
    symbolic_str = (
        f"Hand Description: {finger_description}. "
        f"Palm Orientation Angle: {orientation:.1f} degrees. "
        f"Movement: {motion_type}."
    )
    
    return symbolic_str
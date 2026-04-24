"""
Script to determine the hand state (finger positions, hand orientation, etc.)
for the iconic gesture recognition into Symbolic Representation.
"""
import numpy as np
import cv2

### === Global for fingers === ###
# FINGER = (TIP, BASE)
THUMB = (4, 2)
INDEX = (8, 5)
MIDDLE = (12, 9)
RING = (16, 13)
PINKY = (20, 17)

WRIST = 0

### === Camera information === ###
## NEED to find the actial focal length of the camera for accurate real-world coordinate conversion
FOCAL_LENGTH = 1.0  # Focal length of the camera

def get_fingers_state(hand_landmarks):
    """
    Determine the hand state based on the fingers positions
    EXTENDED = 1 OR FOLDED = 0
    """
    finger_states = []
    landmarks = hand_landmarks.landmark
    wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])

    # 1. THUMB LOGIC (Unchanged robust distance-based logic)
    thumb_tip = np.array([landmarks[4].x, landmarks[4].y])
    pinky_mcp = np.array([landmarks[17].x, landmarks[17].y])
    thumb_mcp = np.array([landmarks[2].x, landmarks[2].y])
    dist_thumb_to_palm = np.linalg.norm(thumb_tip - pinky_mcp)
    dist_mcp_to_palm = np.linalg.norm(thumb_mcp - pinky_mcp)
    finger_states.append(1 if dist_thumb_to_palm > dist_mcp_to_palm else 0)

    # 2. ROBUST FINGER LOGIC (Distance-based for orientation independence)
    finger_tips = [8, 12, 16, 20]
    pip_joints = [6, 10, 14, 18]

    for tip, pip in zip(finger_tips, pip_joints):
        tip_vec = np.array([landmarks[tip].x, landmarks[tip].y, landmarks[tip].z])
        pip_vec = np.array([landmarks[pip].x, landmarks[pip].y, landmarks[pip].z])
        
        # If tip is further from the wrist than its joint, it is extended
        # This works horizontally, vertically, or diagonally.
        if np.linalg.norm(tip_vec - wrist) > np.linalg.norm(pip_vec - wrist):
            finger_states.append(1)
        else:
            finger_states.append(0)

    return finger_states

def get_hand_orientation(hand_landmarks):
# to test out because i'm not sure if it works as wished 
# if necessary --> check mp_HandGesture function palm_orientation
# not using any cross product perturbs me
# needs to be refined or have the angles in another reference frame (like the camera frame) to be more robust to hand orientation in space
    """
    Determine the hand orientation based on the wrist and middle finger base positions
    if facing UP, DOWN, LEFT, RIGHT, FRONT, BACK and orientation (angle)
    """
    landmarks = hand_landmarks.landmark

    wrist = np.array([landmarks[WRIST].x, landmarks[WRIST].y])
    middle_base = np.array([landmarks[MIDDLE[1]].x, landmarks[MIDDLE[1]].y])

    # Vector from wrist to middle finger base
    vector = middle_base - wrist
    angle = np.arctan2(vector[1], vector[0]) * 180 / np.pi

    return vector, angle


# to add : type of motion -- if STATIONNARY or MOVING
# to add : if MOVING --> discription of the motion (direction, speed, type of motion, etc.)
# import numpy as np
# import cv2

# ### === Global for fingers === ###
# # FINGER = (TIP, BASE)
# THUMB = (4, 2)
# INDEX = (8, 5)
# MIDDLE = (12, 9)
# RING = (16, 13)
# PINKY = (20, 17)

# ### === Camera information === ###
# FOCAL_LENGTH = 1.0  # Focal length of the camera


# # TRIAL METHOD
# def recognize_gesture(hand_landmarks):
#     """
#     Simple gesture detection based on landmark positions.
#     """
#     landmarks = hand_landmarks.landmark

#     def is_finger_extended(finger_tip, finger_base):
#         return (
#             landmarks[finger_tip].y < landmarks[finger_base].y
#         )  # Higher y-value = lower in frame

#     thumb_extended = is_finger_extended(4, 2)
#     index_extended = is_finger_extended(8, 5)
#     middle_extended = is_finger_extended(12, 9)
#     ring_extended = is_finger_extended(16, 13)
#     pinky_extended = is_finger_extended(20, 17)

#     if not any([index_extended, middle_extended, ring_extended, pinky_extended]):
#         return "Fist"
#     elif (
#         index_extended and middle_extended and not ring_extended and not pinky_extended
#     ):
#         return "Peace Sign"
#     elif all([index_extended, middle_extended, ring_extended, pinky_extended]):
#         return "Full Palm"
#     return "Unknown"

# def all_fingers_below_palm(finger_tip, finger_base):
#     finger_tip.y
#     THUMB[0].y # y coordinate of tip of thumb

# def project_point(point3D, cx, cy):
#     x, y, z = point3D
#     if z == 0:
#         z = 1e-6  # prevent divide by zero
#     u = int((x * FOCAL_LENGTH / abs(z)) + cx)
#     v = int((y * FOCAL_LENGTH / abs(z)) + cy)
#     return (u, v)


# # FINGER STATE ANALYSIS
# # Function to check if fingers are extended
# def get_finger_states(hand_landmarks):
#     """
#     Returns a list indicating if each finger is extended (1) or folded (0).
#     Uses relative landmark positions instead of just y-coordinates.
#     """
#     finger_states = []

#     # Define landmarks for fingertips and PIP joints
#     finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
#     pip_joints = [6, 10, 14, 18]  # Corresponding PIP joints

#     for tip, pip in zip(finger_tips, pip_joints):
#         if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
#             finger_states.append(1)  # Finger is extended
#         else:
#             finger_states.append(0)  # Finger is folded

#     # Thumb: Check distance instead of y-coordinates
#     thumb_tip = hand_landmarks.landmark[4]
#     thumb_ip = hand_landmarks.landmark[3]
#     thumb_mcp = hand_landmarks.landmark[2]

#     if abs(thumb_tip.x - thumb_mcp.x) > abs(thumb_ip.x - thumb_mcp.x):
#         finger_states.insert(0, 1)  # Thumb extended
#     else:
#         finger_states.insert(0, 0)  # Thumb folded

#     return finger_states


# # Function to recognize gestures based on finger states
# def recognize_gesture2(hand_landmarks, frame):
#     finger_states = get_finger_states(hand_landmarks)
#     print("Finger states: ", finger_states)
#     organic_gesture, cross_vec, coordinates = cross_product_vector(hand_landmarks, frame)
#     # """
#     # Identifies gestures based on which fingers are extended or folded.
#     # """
#     if organic_gesture:
#         ### NOT VERY ROBUST
#         if organic_gesture:#cross_vec:
#             return "Palm Up"
#         elif not organic_gesture:#cross_vec:
#             return "Palm Down"
#     elif not organic_gesture:
#         if finger_states == [1, 1, 1, 1, 1]:
#             return "Open Palm"
#         if finger_states == [0, 0, 0, 0, 0]:
#             return "Fist"
#         elif finger_states == [1, 1, 1, 1, 1]:
#             return "Open Palm"
#         elif finger_states == [0, 1, 1, 0, 0]:
#             return "Peace Sign"
#         elif finger_states == [1, 0, 0, 0, 0]:
#             return "Thumbs Up"
#         elif finger_states == [1, 1, 0, 0, 1]:
#             return "Rock Sign"
#         elif finger_states == [1, 1, 1, 0, 0]:
#             return "Three Fingers"
#         else:
#             return "Unknown Gesture"

import numpy as np
import cv2

### === Global for fingers === ###
# FINGER = (TIP, BASE)
THUMB = (4, 2)
INDEX = (8, 5)
MIDDLE = (12, 9)
RING = (16, 13)
PINKY = (20, 17)

### === Camera information === ###
FOCAL_LENGTH = 1.0  # Focal length of the camera


# TRIAL METHOD
def recognize_gesture(hand_landmarks):
    """
    Simple gesture detection based on landmark positions.
    """
    landmarks = hand_landmarks.landmark

    def is_finger_extended(finger_tip, finger_base):
        return (
            landmarks[finger_tip].y < landmarks[finger_base].y
        )  # Higher y-value = lower in frame

    thumb_extended = is_finger_extended(4, 2)
    index_extended = is_finger_extended(8, 5)
    middle_extended = is_finger_extended(12, 9)
    ring_extended = is_finger_extended(16, 13)
    pinky_extended = is_finger_extended(20, 17)

    if not any([index_extended, middle_extended, ring_extended, pinky_extended]):
        return "Fist"
    elif (
        index_extended and middle_extended and not ring_extended and not pinky_extended
    ):
        return "Peace Sign"
    elif all([index_extended, middle_extended, ring_extended, pinky_extended]):
        return "Full Palm"
    return "Unknown"

def all_fingers_below_palm(finger_tip, finger_base):
    finger_tip.y
    THUMB[0].y # y coordinate of tip of thumb

def project_point(point3D, cx, cy):
    x, y, z = point3D
    if z == 0:
        z = 1e-6  # prevent divide by zero
    u = int((x * FOCAL_LENGTH / abs(z)) + cx)
    v = int((y * FOCAL_LENGTH / abs(z)) + cy)
    return (u, v)


# Function to define the direction of the cross-product vector between the thumb and the fingers
## SHOULD DO THE VECTOR FROM PALM CENTER TO FINGER TIP ##
def cross_product_vector(hand_landmarks, frame):
    height, width, depth = frame.shape
    
    # Extract coordinates
    palm_base = hand_landmarks.landmark[0]  # Palm base (wrist)
    # THUMB
    thumb_base = hand_landmarks.landmark[2]  # Thumb MCP
    thumb_tip = hand_landmarks.landmark[4]   # Thumb TIP

    # INDEX
    finger_base = hand_landmarks.landmark[5]  # Index MCP as base for fingers
    finger_tip = hand_landmarks.landmark[8]   # Index TIP as finger tip

    # Convert to pixel coordinates
    thumb_basePIXEL = [int(thumb_base.x * width), int(thumb_base.y * height), int(thumb_base.z * depth)]
    thumb_tipPIXEL = [int(thumb_tip.x * width), int(thumb_tip.y * height), int(thumb_tip.z * depth)]

    finger_basePIXEL = [int(finger_base.x * width), int(finger_base.y * height), int(finger_base.z * depth)]
    finger_tipPIXEL = [int(finger_tip.x * width), int(finger_tip.y * height), int(finger_tip.z * depth)]
    
    palm_basePIXEL = [int(palm_base.x * width), int(palm_base.y * height), int(palm_base.z * depth)]
    # print("Finger base coordinates: ", finger_basePIXEL)
    # print("Finger tip coordinates: ", finger_tipPIXEL)
    # print("Thumb base coordinates: ", thumb_basePIXEL)
    # print("Thumb tip coordinates: ", thumb_tipPIXEL)
    # print("Palm base coordinates: ", palm_basePIXEL)

    coordinates = [thumb_basePIXEL, thumb_tipPIXEL, finger_basePIXEL, finger_tipPIXEL, palm_basePIXEL]

    # convert to real world coordinates
    cx = width / 2
    cy = height / 2

    thumb_pixel_vect = np.array([thumb_tipPIXEL[0] - palm_basePIXEL[0], thumb_tipPIXEL[1] - palm_basePIXEL[1], thumb_tipPIXEL[2] - palm_basePIXEL[2]])
    finger_pixel_vect = np.array([finger_tipPIXEL[0] - palm_basePIXEL[0], finger_tipPIXEL[1] - palm_basePIXEL[1], finger_tipPIXEL[2] - palm_basePIXEL[2]])
    


    thumb_RW = np.array([((thumb_tip.x * width - cx) * abs(thumb_tip.z) / FOCAL_LENGTH),((thumb_tip.y * width - cy) * abs(thumb_tip.z) / FOCAL_LENGTH), (thumb_tip.z)])
    finger_RW = np.array([((finger_tip.x * width - cx) * abs(finger_tip.z) / FOCAL_LENGTH),((finger_tip.y * width - cy) * abs(finger_tip.z) / FOCAL_LENGTH), (finger_tip.z)])
    palm_RW = np.array([((palm_base.x * width - cx) * abs(palm_base.z) / FOCAL_LENGTH),((palm_base.y * width - cy) * abs(palm_base.z) / FOCAL_LENGTH), (palm_base.z)])

    print("Thumb coordinates: ", thumb_RW, thumb_pixel_vect)
    print("Finger coordinates: ", finger_RW, finger_pixel_vect)
    print("Palm coordinates: ", palm_RW)

    thumb_vec3D = thumb_RW - palm_RW
    finger_vec3D = finger_RW - palm_RW
    cross_vec3D = np.cross(thumb_vec3D, finger_vec3D)
    cross_vec3D_pixel = np.array([cross_vec3D[0] * width, cross_vec3D[1] * height, cross_vec3D[2] * depth])
    print("Cross product vector 3D pixel: ", cross_vec3D_pixel)
    cross_vec2D = project_point(cross_vec3D, cx, cy)
    print("Cross product vector 3D: ", cross_vec3D)
    print("Cross product vector 2D: ", cross_vec2D)
    # # Draw lines for visualization
    # cv2.line(frame, tuple(palm_basePIXEL), tuple(thumb_tipPIXEL), (0, 255, 0), 2)
    # cv2.line(frame, tuple(palm_basePIXEL), tuple(finger_tipPIXEL), (255, 0, 0), 2)

    # # Define vectors
    # thumb_vec = np.array([
    #     thumb_tipPIXEL[0] - palm_basePIXEL[0],
    #     thumb_tipPIXEL[1] - palm_basePIXEL[1],
    #     thumb_tipPIXEL[2] - palm_basePIXEL[2]])
    # finger_vec = np.array([
    #     finger_tipPIXEL[0] - palm_basePIXEL[0],
    #     finger_tipPIXEL[1] - palm_basePIXEL[1],
    #     finger_tipPIXEL[2] - palm_basePIXEL[2]])
    # # thumb_vec = np.array([
    # #     # thumb_tip.x - thumb_base.x,
    # #     # thumb_tip.y - thumb_base.y,
    # #     # thumb_tip.z - thumb_base.z
    # #     thumb_tip.x - palm_base.x,
    # #     thumb_tip.y - palm_base.y,
    # #     thumb_tip.z - palm_base.z
    # # ])

    # # finger_vec = np.array([
    # #     # finger_tip.x - finger_base.x,
    # #     # finger_tip.y - finger_base.y,
    # #     # finger_tip.z - finger_base.z
    # #     finger_tip.x - palm_base.x,
    # #     finger_tip.y - palm_base.y,
    # #     finger_tip.z - palm_base.z
    # # ])

    # print("Thumb vector: ", thumb_vec)
    # print("Finger vector: ", finger_vec)

    # # Compute cross product
    # cross_vec = np.cross(thumb_vec, finger_vec)
    # print("Cross product: ", cross_vec)
    # print("Cross product vector: ", cross_vec)
    # if cross_vec[0] < 0 and cross_vec[1] > 0: #and cross_vec[2] < 0:
    #     return False, False, coordinates# Palm neither Up or Down
    # elif cross_vec[0] > 0 and cross_vec[1] < 0: #and cross_vec[2] < 0:
    #     return True, True, coordinates# Palm Up
    # elif cross_vec[0] < 0 and cross_vec[1] > 0: #and cross_vec[2] > 0:
    #     return True, False, coordinates # Palm Down
    # else: 
    return False, cross_vec2D, coordinates
    # if cross_vec > 0:
    #     return False
    # elif cross_vec < 0:
    #     return True

    # return cross_vec

# FINGER STATE ANALYSIS
# Function to check if fingers are extended
def get_finger_states(hand_landmarks):
    """
    Returns a list indicating if each finger is extended (1) or folded (0).
    Uses relative landmark positions instead of just y-coordinates.
    """
    finger_states = []

    # Define landmarks for fingertips and PIP joints
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
    pip_joints = [6, 10, 14, 18]  # Corresponding PIP joints

    for tip, pip in zip(finger_tips, pip_joints):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            finger_states.append(1)  # Finger is extended
        else:
            finger_states.append(0)  # Finger is folded

    # Thumb: Check distance instead of y-coordinates
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    thumb_mcp = hand_landmarks.landmark[2]

    if abs(thumb_tip.x - thumb_mcp.x) > abs(thumb_ip.x - thumb_mcp.x):
        finger_states.insert(0, 1)  # Thumb extended
    else:
        finger_states.insert(0, 0)  # Thumb folded

    return finger_states


# Function to recognize gestures based on finger states
def recognize_gesture2(hand_landmarks, frame):
    finger_states = get_finger_states(hand_landmarks)
    print("Finger states: ", finger_states)
    organic_gesture, cross_vec, coordinates = cross_product_vector(hand_landmarks, frame)
    # """
    # Identifies gestures based on which fingers are extended or folded.
    # """
    if organic_gesture:
        ### NOT VERY ROBUST
        if organic_gesture:#cross_vec:
            return "Palm Up"
        elif not organic_gesture:#cross_vec:
            return "Palm Down"
    elif not organic_gesture:
        if finger_states == [1, 1, 1, 1, 1]:
            return "Open Palm"
        if finger_states == [0, 0, 0, 0, 0]:
            return "Fist"
        elif finger_states == [1, 1, 1, 1, 1]:
            return "Open Palm"
        elif finger_states == [0, 1, 1, 0, 0]:
            return "Peace Sign"
        elif finger_states == [1, 0, 0, 0, 0]:
            return "Thumbs Up"
        elif finger_states == [1, 1, 0, 0, 1]:
            return "Rock Sign"
        elif finger_states == [1, 1, 1, 0, 0]:
            return "Three Fingers"
        else:
            return "Unknown Gesture"
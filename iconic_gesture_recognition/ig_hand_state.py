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

class HandState:
    def __init__(self, hand_landmarks, frame):
        self.gesture_history = []  # Store recent hand states for temporal analysis
        self.wrist_history = []    # Store recent wrist positions for motion analysis

        self.landmarks = hand_landmarks
        self.label = None

        ### === Camera information === ###
        self.frame = frame
        self.H = frame.shape[0] # Height
        self.W = frame.shape[1] # Width
        self.D = frame.shape[2] # Depth

        self.FOV = 1.32  # radians
        self.FOCAL_LENGTH = self.W / (2 * np.tan(self.FOV / 2))  # Focal length of the camera
        self.centerFrame = (self.W // 2, self.H // 2)

        self.camera_view = np.array([0, 0, -1])

    def get_label(self):
        return self.label

    def project_point(self,point3D, cx, cy):
        x, y, z = point3D
        if z == 0:
            z = 1e-6  # prevent divide by zero
        u = int((x * FOCAL_LENGTH / abs(z)) + cx)
        v = int((y * FOCAL_LENGTH / abs(z)) + cy)
        return (u, v)

    def to_pixel_coords(self, hand_landmarks):
        self.landmarks = hand_landmarks#.landmark
        return int(self.landmarks.x * self.W), int(self.landmarks.y * self.H), int(self.landmarks.z * self.D)
        
    
### === Hand State Functions === ###
    def get_fingers_state(self, hand_landmarks):
        """
        Determine the hand state based on the fingers positions
        EXTENDED = 1 OR FOLDED = 0
        returns a list indicating if each finger is extended (1) or folded (0)
        [THUMB, INDEX, MIDDLE, RING, PINKY]
        """
        finger_states = []
        self.landmarks = hand_landmarks.landmark
        wrist = np.array([self.landmarks[0].x, self.landmarks[0].y, self.landmarks[0].z])

        # 1. THUMB LOGIC (Unchanged robust distance-based logic)
        thumb_tip = np.array([self.landmarks[4].x, self.landmarks[4].y])
        pinky_mcp = np.array([self.landmarks[17].x, self.landmarks[17].y])
        thumb_mcp = np.array([self.landmarks[2].x, self.landmarks[2].y])
        dist_thumb_to_palm = np.linalg.norm(thumb_tip - pinky_mcp)
        dist_mcp_to_palm = np.linalg.norm(thumb_mcp - pinky_mcp)
        finger_states.append(1 if dist_thumb_to_palm > dist_mcp_to_palm else 0)

        # 2. ROBUST FINGER LOGIC (Distance-based for orientation independence)
        finger_tips = [8, 12, 16, 20]
        pip_joints = [6, 10, 14, 18]

        for tip, pip in zip(finger_tips, pip_joints):
            tip_vec = np.array([self.landmarks[tip].x, self.landmarks[tip].y, self.landmarks[tip].z])
            pip_vec = np.array([self.landmarks[pip].x, self.landmarks[pip].y, self.landmarks[pip].z])
            
            # If tip is further from the wrist than its joint, it is extended
            # This works horizontally, vertically, or diagonally.
            if np.linalg.norm(tip_vec - wrist) > np.linalg.norm(pip_vec - wrist):
                finger_states.append(1)
            else:
                finger_states.append(0)

        return finger_states

### === Hand Orientation Functions === ###
    def get_hand_orientation(self, hand_landmarks):
    # to test out because i'm not sure if it works as wished 
    # if necessary --> check mp_HandGesture function palm_orientation
    # not using any cross product perturbs me
    # needs to be refined or have the angles in another reference frame (like the camera frame) to be more robust to hand orientation in space
        """
        Determine the hand orientation based on the wrist and middle finger base positions
        if facing UP, DOWN, LEFT, RIGHT, FRONT, BACK and orientation (angle)
        """
        # self.landmarks = hand_landmarks.landmark

        # wrist = np.array([self.landmarks[WRIST].x, self.landmarks[WRIST].y])
        # middle_base = np.array([self.landmarks[MIDDLE[1]].x, self.landmarks[MIDDLE[1]].y])

        # # Vector from wrist to middle finger base
        # vector = middle_base - wrist
        # angle = np.arctan2(vector[1], vector[0]) * 180 / np.pi

        # return vector, angle
        """
        Function to calculate the orientation of the hand
        """
        self.landmarks = hand_landmarks.landmark
        cross, _, _, _ = self.cross_product(hand_landmarks)

        dot = np.dot(cross, self.camera_view)
        dot = np.clip(dot, -1.0, 1.0)
        angle_rad = np.arccos(dot)
        angle_deg = np.degrees(angle_rad)
        ### Not as robust as needed maybe, the orientation seems a bit off
        
        # print("Angle in degrees: ", angle_deg)

        return angle_deg

    # to add : type of motion -- if STATIONNARY or MOVING --> in the ig_temporal_gesture manager
    # to add : if MOVING --> discription of the motion (direction, speed, type of motion, etc.)


    

    # Function to define the direction of the cross-product vector between the thumb and the fingers
    def cross_product(self, hand_landmarks):
            landmarks = hand_landmarks.landmark

            palm = landmarks[0]
            thumb_tip = landmarks[4]

            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            ring_tip = landmarks[16]
            pinky_tip = landmarks[20]

            # Convert to pixel coordinates
            thumb_tipPIXEL = self.to_pixel_coords(thumb_tip)

            index_tipPIXEL = self.to_pixel_coords(index_tip)
            middle_tipPIXEL = self.to_pixel_coords(middle_tip)
            ring_tipPIXEL = self.to_pixel_coords(ring_tip)
            pinky_tipPIXEL = self.to_pixel_coords(pinky_tip)

            fingers_PIXEL = (index_tipPIXEL + middle_tipPIXEL + ring_tipPIXEL + pinky_tipPIXEL)

            palm_PIXEL = self.to_pixel_coords(palm)
            coordinates = [thumb_tipPIXEL, index_tipPIXEL, middle_tipPIXEL, ring_tipPIXEL, pinky_tipPIXEL, palm_PIXEL]#, fingers_PIXEL]

            # vectors for cross product
            vthumb = np.array([thumb_tip.x - palm.x, thumb_tip.y - palm.y, thumb_tip.z - palm.z])
            vindex = np.array([index_tip.x - palm.x, index_tip.y - palm.y, index_tip.z - palm.z])
            vmiddle = np.array([middle_tip.x - palm.x, middle_tip.y - palm.y, middle_tip.z - palm.z])
            vring = np.array([ring_tip.x - palm.x, ring_tip.y - palm.y, ring_tip.z - palm.z])
            vpinky = np.array([pinky_tip.x - palm.x, pinky_tip.y - palm.y, pinky_tip.z - palm.z])

            vfingers = (vindex+vmiddle+vring+vpinky)/4
            cross = np.cross(vthumb, vfingers)

            if self.label == 'Right':
                # print("Right hand")
                cross = cross

            elif self.label == 'Left':
                # print("Left hand")
                cross = -cross

            cross /= np.linalg.norm(cross)

            scale = 500

            end_px = (int(palm_PIXEL[0] + cross[0] * scale),
                    int(palm_PIXEL[1] + cross[1] * scale))
            
            # end_px2 = (int(palm_PIXEL[0] + cross2[0] *scale),
                    #   int(palm_PIXEL[1] + cross2[1] *scale))
            
            return cross, coordinates, palm_PIXEL, end_px#, end_px2

    def draw_cross_product_vector(self, frame, hand_landmarks):
        cross, coordinates, palm_px, end_px = self.cross_product(hand_landmarks)
        thumb_PIXEL = coordinates[0]
        index_PIXEL = coordinates[1]
        middle_PIXEL =  coordinates[2]
        ring_PIXEL =  coordinates[3]
        pinky_PIXEL =  coordinates[4]
        palm_PIXEL =  coordinates[5]


        cv2.line(frame, tuple(palm_px[:2]), end_px, (0,0,255), 2) # Cross Product
        # cv2.line(frame, tuple(palm_px[:2]), end_px2, (0,0,0), 2) # Cross Product
        # cv2.line(frame, palm_pxRW, end_pxRW, (255,255,255), 3) # Cross Product RW
        cv2.line(frame, tuple(palm_PIXEL[:2]), tuple(thumb_PIXEL[:2]), (0, 255, 0), 2)
        cv2.line(frame, tuple(palm_PIXEL[:2]), tuple(middle_PIXEL[:2]), (255, 0, 0), 2)
        # cv2.line(frame, tuple(palm_PIXEL[:2]), tuple(fingers_PIXEL[:2]), (255, 0, 0), 2)
"""
Script to determine the hand state (finger positions, hand orientation, etc.)
for the iconic gesture recognition into Symbolic Representation.
"""
import numpy as np
import cv2

### === Global for fingers === ###
# FINGER = (BASE, PIP, DIP, TIP)
# !!!! Change in landmark indexing for all fingers !!!!
THUMB = (1, 2, 3, 4)
INDEX = (5, 6, 7, 8)
MIDDLE = (9, 10, 11, 12)
RING = (13, 14, 15, 16)
PINKY = (17, 18, 19, 20)

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


    # set of 6 rules to encode hand gesture in terms of finger flexion, proximity, contact, direction
    # pal, orientation and hand position based on paper of GestureGPT

    def vector_angle(self, v1, v2):
        """Helper function to calculate the angle between two vectors."""
        unit_v1 = v1 / (np.linalg.norm(v1) + 1e-6)  # Avoid division by zero
        unit_v2 = v2 / (np.linalg.norm(v2) + 1e-6)

        dot_product = np.dot(unit_v1, unit_v2)

        angle = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))

        return angle
    
    def get_landmark_vector(self, lm):
        """Convert a landmark to a 3D vector."""
        return np.array([lm.x, lm.y, lm.z])

    def finger_flexion(self, hand_landmarks, finger_type, th_low=60, th_high=75):
        # the thresholds are the sum of the angles --> needs to be finetuned 
        # otherwise function works
        # it can also give information on the amount of fexion of the finger
        # and not just the binary information of extended or folded
        """Rule 1: Figer Flexion
        Computes if a finger is straight (1), in between (0) or folded (0)
        by summing the bending angles of its joints
        """
        self.landmarks = hand_landmarks.landmark
        lm = self.landmarks
        curl = 0.0

        # works okey but thumb flexion is tricky, maybe the first version is better for
        # the thumb even though i like this method better for the other fingers,
        # maybe just need to reajust the thresholds
        if finger_type == 'THUMB':
            th_low = 15
            th_high = 40
            # Vectors: MCP->IP and IP->TIP
            v1 = self.get_landmark_vector(lm[3]) - self.get_landmark_vector(lm[2])
            v2 = self.get_landmark_vector(lm[4]) - self.get_landmark_vector(lm[3])
            curl = self.vector_angle(v1, v2)

        else:
            # dictionnary mapping finger names to their landmark/joint indices 
            joints = {'INDEX': INDEX, 'MIDDLE': MIDDLE, 
                      'RING' :RING, 'PINKY': PINKY}
            base, pip, dip, tip = joints[finger_type]

            # angle 1: base->pip and pip->dip
            v1 = self.get_landmark_vector(lm[pip]) - self.get_landmark_vector(lm[base])
            v2 = self.get_landmark_vector(lm[dip]) - self.get_landmark_vector(lm[pip])
            curl += self.vector_angle(v1, v2) # because computing based on angle sum

            # angle 2: pip->dip and dip->tip
            v3 = self.get_landmark_vector(lm[tip]) - self.get_landmark_vector(lm[dip])
            curl += self.vector_angle(v2, v3)

            # print(f"{finger_type} curl: {curl:.2f} ")

        if curl <= th_low:
            return 1  # Straight
        elif curl >= th_high:
            return -1  # Bent
        else:
            return 0  # In between or unsure
        
    def get_finger_flexion(self, hand_landmarks,th_low=60, th_high=75):
        """
        have a function that computes the finger flexions in the same way as finger_fexion()
        but returns the states as the function get_finger_state()
        """
        self.landmarks = hand_landmarks.landmark
        curl = 0.0
        finger_states = []


    def finger_proximity(self, hand_landmarks, f1_idx, f2_idx, th_low=0.03, th_high=0.06):
        # needs from refinment because so far the threasholds will depend on the distance of the 
        # hand to the camera
        # maybe need to add a normalization factor based on the depth of the hand to make it more robust to distance changes
        # figure out the entire usefulness of this rule (if really necessary)

        # doesn't seem to be a very robust information to be used

        """
        Rule 2: Finger Proximity
        Computes if fingers are pressed to each other (1), apart (-1), or in between (0)
        by checking the average minimal distance between joints
        """
        # Extract the joint landmarks for the two fingers
        f1_joints = [self.get_landmark_vector(hand_landmarks.landmark[idx]) for idx in f1_idx]
        f2_joints = [self.get_landmark_vector(hand_landmarks.landmark[idx]) for idx in f2_idx]

        # Compute distances between all pairs of joints
        distances = []
        for j1 in f1_joints:
            for j2 in f2_joints:
                dist = np.linalg.norm(j1 - j2)
                distances.append(dist)

        min_dist = np.min(distances) # using just the minimal distance, not the avg min distance

        if min_dist <= th_low:
            return 1  # Pressed together
        elif min_dist >= th_high:
            return -1  # Apart
        else:
            return 0  # In between or unsure

    def finger_contact(self, hand_landmarks, target_tip_idx, th_low=0.04, th_high=0.06):
        # Works well, but has some issues when the contact is facing the camera
        # it can't distinguish the tip landmaks position properly
        """
        Rule 3: Finger Contact
        Compute the Euclidean distance between the thumb tip and another finger tip
        """
        self.landmarks = hand_landmarks.landmark

        thumb_tip = self.get_landmark_vector(self.landmarks[THUMB[3]])  # Thumb tip
        finger_tip = self.get_landmark_vector(self.landmarks[target_tip_idx])  # Target finger tip

        dist = np.linalg.norm(thumb_tip - finger_tip)

        if dist <= th_low:
            return 1  # In contact
        elif dist >= th_high:
            return -1  # Not in contact
        else:
            return 0  # In between or unsure

    def thumb_direction(self, hand_landmarks, is_thumb_straight, th=40):
        # for now, large threashold but works well
        """
        Rule 4: Thumb Pointing Direction    # to be extended to other fingers ?
        Finds the thumb direction vector against world-axis vectors
        only applies when the thumb is straight
        """
        self.landmarks = hand_landmarks.landmark

        if not is_thumb_straight:
            return 0  # Not applicable if thumb is not straight

        # Thumb vector from the MCP/PIP to the tipe
        thumb_vec = self.get_landmark_vector(self.landmarks[THUMB[3]]) - self.get_landmark_vector(self.landmarks[THUMB[1]])
        
        # define reference direction 
        # (Y is down in OpenCV/MediaPipe image coordinates ==> -Y is up)
        # This may need to be changed with Robot camera !!!
        up_vec = np.array([0, -1, 0])
        down_vec = np.array([0, 1, 0])

        angle_up = self.vector_angle(thumb_vec, up_vec)
        angle_down = self.vector_angle(thumb_vec, down_vec)

        if angle_up < th and angle_up < angle_down:
            return 1 # Upwards
        elif angle_down < th and angle_down < angle_up:
            return -1 # Downwards
        else:
            return 0 # In between or unsure


    def palm_orientation(self, frame, hand_landmarks, hand_label, th=30):
        # new way to compute, hopefully more robust
        """
        Rule 5: Palm Orientation
        Compute the cross product of 2 vectors ON the palm to get the palm orientation
        """
        self.landmarks = hand_landmarks.landmark
        
        # Palm vector 1: Pinky base to Index base
        palm_vec1 = self.get_landmark_vector(self.landmarks[PINKY[0]]) - self.get_landmark_vector(self.landmarks[INDEX[0]])
        # Palm vector 2: wrist to Middle base
        palm_vec2 = self.get_landmark_vector(self.landmarks[WRIST]) - self.get_landmark_vector(self.landmarks[MIDDLE[0]])


        if hand_label == 'Left':
            palm_normal = np.cross(palm_vec1, palm_vec2)
        else:
            palm_normal = np.cross(palm_vec2, palm_vec1)

        palm_normal = palm_normal / (np.linalg.norm(palm_normal) + 1e-6)  # Normalize

        # Define reference vectors
        directions = {
            # 'Up': np.array([0, -1, 0]),
            'Up': np.array([0, 1, 0]),
            # 'Down': np.array([0, 1, 0]),  # i think this is the correct orientation
            'Down': np.array([0, -1, 0]),
            'Left': np.array([-1, 0, 0]),
            'Right': np.array([1, 0, 0]),
            'Inward': np.array([0, 0, 1]),   # Pointing towards camera
            'Outward': np.array([0, 0, -1])  # Pointing away from camera
        }

        wrist_px = (int(self.landmarks[WRIST].x * self.W), int(self.landmarks[WRIST].y * self.H))
        end_px = (int(wrist_px[0] + palm_normal[0] * -100), int(wrist_px[1] + palm_normal[1] * -100))

        # trace the line of the cross product vector for visualization, from the wrist
        cv2.line(frame, wrist_px, end_px, (255, 0, 0), 2)
        
        min_angle = float('inf')
        best_dir = 'Unknown'
        
        for name, ref_vec in directions.items():
            angle = self.vector_angle(palm_normal, ref_vec)
            if angle < min_angle:
                min_angle = angle
                best_dir = name
                
        if min_angle > th:
            return 'Unknown'
        return best_dir



    def hand_position(self, hand_landmarks):
        # previously had the hand displacement based on the center of the camera,
        # check to see if we should combine, if necessary to add or replace
        """
        Rule 6: Hand Position
        Simply the Geometric center of the hand to track general movement over time
        """
        self.landmarks = hand_landmarks.landmark
        
        all_points = np.array([self.get_landmark_vector(pt) for pt in self.landmarks])
        center = np.mean(all_points, axis=0)
        return center.tolist() # returns coordinates [x, y, z] with respect to the camera frame, can be used to track hand movement over time and detect if the hand is moving towards or away from the camera, or moving left/right/up/down in the camera view. This can be useful for temporal gesture recognition and motion analysis.
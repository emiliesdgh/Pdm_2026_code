"""
Script to determine the hand state (finger positions, hand orientation, etc.)
for the iconic gesture recognition into Symbolic Representation.
"""
import numpy as np
import cv2

### === Global for fingers === ###
FINGERS ={
    "name": ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"],
    "tip_idx": [4, 8, 12, 16, 20],
    "dip_idx": [3, 7, 11, 15, 19],
    "pip_idx": [2, 6, 10, 14, 18],
    "base_idx": [1, 5, 9, 13, 17]
}
WRIST = 0

### === Camera information === ###
## NEED to find the actial focal length of the camera for accurate real-world coordinate conversion
# will need to do a camera calibration process to get the actual focal length and other intrinsic parameters for accurate real-world coordinate conversion
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

### === Helper Functions === ###
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
        # self.landmarks = hand_landmarks#.landmark
        return int(hand_landmarks.x * self.W), int(hand_landmarks.y * self.H), int(hand_landmarks.z * self.D)
        
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
    

### === Finger State Function -- Extended, Folded or inbetween === ###    
    def finger_flexion(self, hand_landmarks, finger_type, th_low=50, th_high=120):
        """Rule 1: Figer Flexion
        Computes if a finger is straight (1), in between (0) or folded (0)
        by summing the bending angles of its joints

        Adding the distance based version of previous version for added robustness (expecially for the thumb)
        this can avoir hard setting thresholds that could vary for users
        distance based method is also orientation independent
        it can be the condition for folded, while the angle can be the condition for straight, and in between if one of them is in between
        """
        self.landmarks = hand_landmarks.landmark
        curl = 0.0
        folded = False
        wrist_vect = self.get_landmark_vector(self.landmarks[WRIST])


        if finger_type == 'THUMB':
            thumb_tip_vect = self.get_landmark_vector(self.landmarks[FINGERS["tip_idx"][0]])  # Thumb tip
            pinky_base_vect = self.get_landmark_vector(self.landmarks[FINGERS["base_idx"][4]])  # Pinky base
            thumb_pip_vect = self.get_landmark_vector(self.landmarks[FINGERS["pip_idx"][0]])  # Thumb pip
            dist_thumb_palm = np.linalg.norm(thumb_tip_vect - pinky_base_vect)
            dist_pip_palm = np.linalg.norm(thumb_pip_vect - pinky_base_vect)
            if dist_thumb_palm > dist_pip_palm:
                return 1  # Folded
            else:
                return -1  # Extended

        else:
            # angle 1: base->pip and pip->dip
            v1 = self.get_landmark_vector(self.landmarks[FINGERS["pip_idx"][FINGERS["name"].index(finger_type)]]) - self.get_landmark_vector(self.landmarks[FINGERS["base_idx"][FINGERS["name"].index(finger_type)]])
            v2 = self.get_landmark_vector(self.landmarks[FINGERS["dip_idx"][FINGERS["name"].index(finger_type)]]) - self.get_landmark_vector(self.landmarks[FINGERS["pip_idx"][FINGERS["name"].index(finger_type)]])
            curl += self.vector_angle(v1, v2) # because computing based on angle sum

            # angle 2: pip->dip and dip->tip
            v3 = self.get_landmark_vector(self.landmarks[FINGERS["tip_idx"][FINGERS["name"].index(finger_type)]]) - self.get_landmark_vector(self.landmarks[FINGERS["dip_idx"][FINGERS["name"].index(finger_type)]])
            curl += self.vector_angle(v2, v3)

            # Distance based condition
            tip_vect = self.get_landmark_vector(self.landmarks[FINGERS["tip_idx"][FINGERS["name"].index(finger_type)]]) 
            pip_vect = self.get_landmark_vector(self.landmarks[FINGERS["pip_idx"][FINGERS["name"].index(finger_type)]])
            if np.linalg.norm(tip_vect - wrist_vect) > np.linalg.norm(pip_vect - wrist_vect):
                folded = True

            # print(f"{finger_type} curl angle: {curl}")
            

        if curl <= th_low:
            return 1  # Extended
        elif curl >= th_high or folded:
            return -1  # Folded
        else:
            return 0  # In between or unsure
        
    def get_finger_flexion_state(self, hand_landmarks,th_low=60, th_high=75):
        """
        Returns a list of finger flexion states using finger_flexion(): [THUMB, INDEX, MIDDLE, RING, PINKY]
        Values:
        1  -> Extended
        0  -> In-between
       -1  -> Folded
        """
        self.landmarks = hand_landmarks.landmark
        finger_types = FINGERS["name"]
        finger_state = []

        for finger in finger_types:
            state = self.finger_flexion(hand_landmarks, finger, th_low, th_high)
            finger_state.append(state)

        return finger_state

### === Hand Orientation to camera & Position to center(?) === ###
    def hand_orientation(self, frame, hand_landmarks, hand_label, th=45):
        # new way to compute, hopefully more robust
        """
        Rule 5: Palm Orientation
        Compute the cross product of 2 vectors ON the palm to get the palm orientation
        the threshold of 45° makes the best direction description in my opinion
        """
        self.landmarks = hand_landmarks.landmark
        
        # Palm vector 1: Pinky base to Index base
        palm_vec1 = self.get_landmark_vector(self.landmarks[FINGERS["base_idx"][4]]) - self.get_landmark_vector(self.landmarks[FINGERS["base_idx"][1]])
        # Palm vector 2: wrist to Middle base
        palm_vec2 = self.get_landmark_vector(self.landmarks[WRIST]) - self.get_landmark_vector(self.landmarks[FINGERS["base_idx"][2]])


        if hand_label == 'Left':
            palm_normal = np.cross(palm_vec1, palm_vec2)
        else:
            palm_normal = np.cross(palm_vec2, palm_vec1)

        palm_normal = palm_normal / (np.linalg.norm(palm_normal) + 1e-6)  # Normalize

        # Define reference vectors
        directions = {
            'Up': np.array([0, 1, 0]),
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

        # -- computation of the angle of the hand orientation
        ## Need to confirm computation is correct
        dot = np.dot(palm_normal, self.camera_view)
        dot = np.clip(dot, -1.0, 1.0)  # Clip for numerical stability
        angle_rad = np.arccos(dot)
        angle_deg = np.degrees(angle_rad)
                
        if min_angle > th:
            return 'Unknown', angle_deg
        return best_dir, angle_deg
    
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

        centerHand = self.landmarks[WRIST]
        centerHand_px = self.to_pixel_coords(centerHand)

        # Displacement from the center of the camera view
        dx = self.centerFrame[0] - centerHand_px[0]
        dy = self.centerFrame[1] - centerHand_px[1]

        # Real Camera coordinates
        # Left/Right : - = left from center point, + = right from center point
        x_realCam = dx * self.D / self.FOCAL_LENGTH
        # Up/Down : - = up from center point, + = down from center point
        y_realCam = dy * self.D / self.FOCAL_LENGTH

        return center.tolist(), (x_realCam, y_realCam)  # to verify the units of (x_realCam, y_realCam) and how to use this information, if necessary at all
    
        # returns coordinates [x, y, z] with respect to the camera frame, 
        # can be used to track hand movement over time and detect if the hand 
        # is moving towards or away from the camera, or moving left/right/up/down 
        # in the camera view. This can be useful for temporal gesture recognition
        # and motion analysis.

### === Other rules for symbolic representation === ###
    def finger_contact(self, hand_landmarks, target_tip_idx, th_low=0.045, th_high=0.07):
        # Works well, but has some issues when the contact is facing the camera
        # it can't distinguish the tip landmaks position properly
        """
        Rule 3: Finger Contact
        Compute the Euclidean distance between the thumb tip and another finger tip
        """
        self.landmarks = hand_landmarks.landmark

        thumb_tip_vect = self.get_landmark_vector(self.landmarks[FINGERS["tip_idx"][0]])  # Thumb tip
        finger_tip_vect = self.get_landmark_vector(self.landmarks[target_tip_idx])  # Target finger tip

        dist = np.linalg.norm(thumb_tip_vect - finger_tip_vect)
        # print(f"Distance between thumb tip and finger tip (idx {target_tip_idx}): {dist}")

        if dist <= th_low:
            return 1  # In contact
        elif dist >= th_high:
            return -1  # Not in contact
        else:
            return 0  # In between or unsure
        
    def get_finger_contact_state(self, hand_landmarks):
        """
        Returns a list of finger contact states using finger_contact(): [THUMB, INDEX, MIDDLE, RING, PINKY]
        Values:
        1  -> In contact
        0  -> In between
       -1  -> Not in contact
        """
        # self.landmarks = hand_landmarks.landmark
        contact_state = []

        for i, finger in enumerate(FINGERS["name"][1:], start=1):
            tip_idx = FINGERS["tip_idx"][i]
            state = self.finger_contact(hand_landmarks, target_tip_idx=tip_idx)
            contact_state.append(state)

        return contact_state


### === Additional hand state rules for symbolic representation -- too see if will be used === ###

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
        thumb_vec = self.get_landmark_vector(self.landmarks[FINGERS["tip_idx"][0]]) - self.get_landmark_vector(self.landmarks[FINGERS["pip_idx"][0]])
        
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



    
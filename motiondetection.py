import cv2
import mediapipe as mp
import numpy as np
import socket
import time

class PoseDetector:
    def __init__(self):
        # Khởi tạo MediaPipe cho Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Khởi tạo MediaPipe cho Hand
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Drawing utils
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Khởi tạo UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.unity_address = ('127.0.0.1', 5000)
        
        # Các điểm landmark quan trọng cho pose
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_ELBOW = 13
        self.RIGHT_ELBOW = 14
        self.LEFT_WRIST = 15
        self.RIGHT_WRIST = 16
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24
        self.LEFT_KNEE = 25
        self.RIGHT_KNEE = 26
        self.LEFT_ANKLE = 27
        self.RIGHT_ANKLE = 28
        self.NOSE = 0
        self.LEFT_EYE = 2
        self.RIGHT_EYE = 5
        
        # Tracking để phát hiện chuyển động ném
        self.prev_right_wrist_pos = None
        self.throw_cooldown = 0
        
        # Tracking cho hand gestures
        self.prev_hand_gesture = None
        self.hand_gesture_cooldown = 0
        
        # Tracking cho các cử chỉ tay đặc biệt
        self.one_finger_cooldown = 0
        self.two_finger_cooldown = 0
        self.open_left_hand_cooldown = 0
        
    def detect_pose(self, pose_landmarks):
        """Nhận diện các pose cơ bản từ landmarks"""
        
        # Lấy các điểm quan trọng
        left_wrist = pose_landmarks[self.LEFT_WRIST]
        right_wrist = pose_landmarks[self.RIGHT_WRIST]
        left_shoulder = pose_landmarks[self.LEFT_SHOULDER]
        right_shoulder = pose_landmarks[self.RIGHT_SHOULDER]
        left_hip = pose_landmarks[self.LEFT_HIP]
        right_hip = pose_landmarks[self.RIGHT_HIP]
        left_knee = pose_landmarks[self.LEFT_KNEE]
        right_knee = pose_landmarks[self.RIGHT_KNEE]
        left_ankle = pose_landmarks[self.LEFT_ANKLE]
        right_ankle = pose_landmarks[self.RIGHT_ANKLE]
        left_elbow = pose_landmarks[self.LEFT_ELBOW]
        right_elbow = pose_landmarks[self.RIGHT_ELBOW]
        nose = pose_landmarks[self.NOSE]
        left_eye = pose_landmarks[self.LEFT_EYE]
        right_eye = pose_landmarks[self.RIGHT_EYE]

        # Detect Turning Left/Right based on head position
        def detect_turning():
            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
            if nose.x < shoulder_center_x - 0.05:
                return "TURNING_RIGHT", abs(nose.x - shoulder_center_x) * 5
            elif nose.x > shoulder_center_x + 0.05:
                return "TURNING_LEFT", abs(nose.x - shoulder_center_x) * 5
            return None, 0

        # Detect Walking (giơ tay trái)
        def is_walking():
            # Kiểm tra nếu cổ tay trái cao hơn vai trái
            if left_wrist.y < left_shoulder.y - 0.1:
                return True, (left_shoulder.y - left_wrist.y) * 5
            return False, 0

        # Detect Throwing (giơ tay phải)
        def is_throwing():
            # Kiểm tra nếu cổ tay phải cao hơn vai phải
            if right_wrist.y < right_shoulder.y - 0.1:
                return True, (right_shoulder.y - right_wrist.y) * 5
            return False, 0

        # Kiểm tra các pose với ưu tiên rõ ràng
        walking, walk_conf = is_walking()
        throwing, throw_conf = is_throwing()
        turning_pose, turn_conf = detect_turning()

        # Kiểm tra các pose khác
        poses = {
            "WALKING": walk_conf,
            "THROWING": throw_conf
        }
        
        # Nếu phát hiện xoay người, ưu tiên pose này
        if turning_pose and turn_conf > 0.2:
            return turning_pose, turn_conf
            
        best_pose = max(poses.items(), key=lambda x: x[1])
        return best_pose if best_pose[1] > 0.2 else ("IDLE", 0)
    
    def detect_hand_gesture(self, hand_landmarks, hand_label):
        """Nhận diện các cử chỉ tay từ landmarks"""
        
        wrist = hand_landmarks[0]
        thumb_cmc = hand_landmarks[1]
        thumb_mcp = hand_landmarks[2]
        thumb_ip = hand_landmarks[3]
        thumb_tip = hand_landmarks[4]
        index_mcp = hand_landmarks[5]
        index_pip = hand_landmarks[6]
        index_dip = hand_landmarks[7]
        index_tip = hand_landmarks[8]
        middle_mcp = hand_landmarks[9]
        middle_pip = hand_landmarks[10]
        middle_dip = hand_landmarks[11]
        middle_tip = hand_landmarks[12]
        ring_mcp = hand_landmarks[13]
        ring_pip = hand_landmarks[14]
        ring_dip = hand_landmarks[15]
        ring_tip = hand_landmarks[16]
        pinky_mcp = hand_landmarks[17]
        pinky_pip = hand_landmarks[18]
        pinky_dip = hand_landmarks[19]
        pinky_tip = hand_landmarks[20]
        
        def is_finger_extended(tip, pip, mcp, threshold=0.05):
            return tip.y < pip.y - threshold
        
        def is_finger_raised(tip, pip, mcp, wrist):
            is_extended_y = tip.y < pip.y and pip.y < mcp.y
            is_higher = tip.y < wrist.y - 0.1
            return is_extended_y or is_higher
        
        def is_thumb_extended(tip, ip, mcp, is_left_hand):
            if is_left_hand:
                return tip.x < ip.x - 0.03
            else:
                return tip.x > ip.x + 0.03
        
        def distance(p1, p2):
            return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
        
        is_left = hand_label == "Left"
        thumb_extended = is_thumb_extended(thumb_tip, thumb_ip, thumb_mcp, is_left)
        index_extended = is_finger_extended(index_tip, index_pip, index_mcp) or is_finger_raised(index_tip, index_pip, index_mcp, wrist)
        middle_extended = is_finger_extended(middle_tip, middle_pip, middle_mcp) or is_finger_raised(middle_tip, middle_pip, middle_mcp, wrist)
        ring_extended = is_finger_extended(ring_tip, ring_pip, ring_mcp) or is_finger_raised(ring_tip, ring_pip, ring_mcp, wrist)
        pinky_extended = is_finger_extended(pinky_tip, pinky_pip, pinky_mcp) or is_finger_raised(pinky_tip, pinky_pip, pinky_mcp, wrist)
        
        finger_states = {
            "thumb": thumb_extended,
            "index": index_extended,
            "middle": middle_extended,
            "ring": ring_extended,
            "pinky": pinky_extended
        }
        
        index_higher_than_wrist = index_tip.y < wrist.y
        middle_higher_than_wrist = middle_tip.y < wrist.y
        
        if not is_left:
            if index_extended and not middle_extended and not ring_extended and not pinky_extended:
                if self.one_finger_cooldown == 0:
                    self.one_finger_cooldown = 15
                    return "RIGHT_ONE_FINGER", 0.95, "Right"
                else:
                    self.one_finger_cooldown = max(0, self.one_finger_cooldown - 1)
            elif index_higher_than_wrist and not middle_higher_than_wrist:
                if self.one_finger_cooldown == 0:
                    self.one_finger_cooldown = 15
                    return "RIGHT_ONE_FINGER", 0.9, "Right"
                else:
                    self.one_finger_cooldown = max(0, self.one_finger_cooldown - 1)
        
        if not is_left:
            if index_extended and middle_extended and not ring_extended and not pinky_extended:
                if self.two_finger_cooldown == 0:
                    self.two_finger_cooldown = 15
                    return "RIGHT_TWO_FINGER", 0.95, "Right"
                else:
                    self.two_finger_cooldown = max(0, self.two_finger_cooldown - 1)
            elif index_higher_than_wrist and middle_higher_than_wrist:
                if self.two_finger_cooldown == 0:
                    self.two_finger_cooldown = 15
                    return "RIGHT_TWO_FINGER", 0.9, "Right"
                else:
                    self.two_finger_cooldown = max(0, self.two_finger_cooldown - 1)
        
        if is_left:
            fingers_extended_count = sum([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended])
            if fingers_extended_count >= 3:
                if self.open_left_hand_cooldown == 0:
                    self.open_left_hand_cooldown = 15
                    return "LEFT_OPEN_HAND", 0.95, "Left"
                else:
                    self.open_left_hand_cooldown = max(0, self.open_left_hand_cooldown - 1)
        
        if not thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return "FIST", 0.9, hand_label
        
        if not is_left and thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended:
            return "OPEN_HAND", 0.9, hand_label
        
        if thumb_extended and index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return "GUN", 0.9, hand_label
        
        if distance(thumb_tip, index_tip) < 0.05:
            return "OK", 0.9, hand_label
        
        if thumb_extended and not index_extended and not middle_extended and not ring_extended and pinky_extended:
            return "CALL", 0.9, hand_label
        
        if thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return "THUMBS_UP", 0.9, hand_label
        
        if not thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            thumb_direction = thumb_tip.y - wrist.y
            if thumb_direction > 0:
                return "THUMBS_DOWN", 0.9, hand_label
        
        self.one_finger_cooldown = max(0, self.one_finger_cooldown - 1)
        self.two_finger_cooldown = max(0, self.two_finger_cooldown - 1)
        self.open_left_hand_cooldown = max(0, self.open_left_hand_cooldown - 1)
        
        return "UNKNOWN", 0.5, hand_label

    def start_detection(self):
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pose_results = self.pose.process(image)
            hand_results = self.hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            final_pose = "IDLE"
            final_confidence = 0
            hand_gesture = None
            hand_gesture_side = None
            finger_debug_info = {}
            
            if pose_results.pose_landmarks:
                pose_name, confidence = self.detect_pose(pose_results.pose_landmarks.landmark)
                if final_pose == "IDLE":
                    final_pose = pose_name
                    final_confidence = confidence
                self.mp_drawing.draw_landmarks(
                    image, 
                    pose_results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
            
            if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                    hand_label = handedness.classification[0].label
                    gesture_name, gesture_conf, gesture_side = self.detect_hand_gesture(
                        [landmark for landmark in hand_landmarks.landmark], 
                        hand_label
                    )
                    
                    if gesture_name == "UNKNOWN":
                        landmarks = hand_landmarks.landmark
                        index_tip_y = landmarks[8].y
                        index_pip_y = landmarks[6].y
                        index_diff = index_pip_y - index_tip_y
                        middle_tip_y = landmarks[12].y
                        middle_pip_y = landmarks[10].y
                        middle_diff = middle_pip_y - middle_tip_y
                        wrist_y = landmarks[0].y
                        index_to_wrist = wrist_y - index_tip_y
                        middle_to_wrist = wrist_y - middle_tip_y
                        
                        finger_debug_info = {
                            "hand": hand_label,
                            "index_diff": f"{index_diff:.3f}",
                            "middle_diff": f"{middle_diff:.3f}",
                            "index_to_wrist": f"{index_to_wrist:.3f}",
                            "middle_to_wrist": f"{middle_to_wrist:.3f}"
                        }
                    
                    if gesture_conf > 0.7:
                        hand_gesture = gesture_name
                        hand_gesture_side = gesture_side
                        if gesture_name == "RIGHT_ONE_FINGER":
                            final_pose = "WALKING"
                            final_confidence = gesture_conf
                        elif gesture_name == "LEFT_OPEN_HAND":
                            final_pose = "THROWING"
                            final_confidence = gesture_conf
                        else:
                            hand_pose = f"HAND_{gesture_name}"
                            if final_pose == "IDLE":
                                final_pose = hand_pose
                                final_confidence = gesture_conf
                    
                    self.mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
            
            data = f"{final_pose},{final_confidence:.2f}".encode()
            self.sock.sendto(data, self.unity_address)
            
            cv2.putText(image, f"Pose: {final_pose}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"Confidence: {final_confidence:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if hand_gesture:
                cv2.putText(image, f"Hand gesture: {hand_gesture} ({hand_gesture_side})", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if finger_debug_info:
                y_pos = 150
                for key, value in finger_debug_info.items():
                    cv2.putText(image, f"{key}: {value}", (10, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    y_pos += 25
            
            cv2.imshow('MediaPipe Pose & Hand Detection', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = PoseDetector()
    detector.start_detection()
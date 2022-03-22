import copy
from collections import Counter
from collections import deque
import cv2 as cv
import mediapipe as mp
from helper import *

cam_width = 960
cam_height = 540
min_detection_confidence  = 0.7
min_tracking_confidence = 0.7
mpDraw = mp.solutions.drawing_utils

previous_action_id = 0
current_action_id = 0
auto_mode = False
light_mode = False
def main():

    global previous_action_id
    global current_action_id
    global auto_mode
    global light_mode
    
    cap_device = 0
    cap_width = cam_width
    cap_height = cam_height

    use_static_image_mode = False
    use_brect = True

    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()
    #point_history_classifier = PointHistoryClassifier()

    action_list = ["stop", "forward", "backward", "go left", "go right", 
    "spin right", "spin left", "forward faster", "do nothing", "change mode", "change light"]
    keypoint_classifier_labels = ["upward fist", "normal fist", "reverse fist", "palm", 
    "reverse palm", "thumb left", "thumb right", "OK", "index up"]
    while True:

        ##################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1) 
        debug_image = copy.deepcopy(image)

        ##############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Calculation of rectangle boundaries
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                cv.putText(debug_image, "Detected shape: " + keypoint_classifier_labels[hand_sign_id], 
                (10,30), 
                cv.FONT_HERSHEY_DUPLEX, 
                0.5, (0, 0, 0), 1, cv.LINE_AA)
                previous_action_id = current_action_id
                current_action_id = getActionAndMode(hand_sign_id, hand_landmarks.landmark[0].x)

                ## Draw result
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_label(
                    debug_image,
                    brect,
                    handedness,
                    action_list[current_action_id],
                    #keypoint_classifier_labels[hand_sign_id],
                    auto_mode
                )
        else:
            current_action_id = 8 # do nothing
            cv.putText(debug_image, "Detected shape: None", 
                (10,30), 
                cv.FONT_HERSHEY_DUPLEX, 
                0.5, (0, 0, 0), 1, cv.LINE_AA)

        current_mode = "Auto" if auto_mode else "Manual"
        current_light = "On" if light_mode else "Off"
        cv.putText(debug_image, "Current Mode: " + current_mode, (10,60), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
        cv.putText(debug_image, "Car Light: " + current_light, (10,90), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

        # SEND TO PI ThE CURRENT_ACTION_ID
        #if not auto_mode:


        ####################################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()

def getActionAndMode(hand_sign_id, hand_place):
    global auto_mode
    global light_mode
    global previous_action_id
    if hand_sign_id == 0 or hand_sign_id == 1: # UPWARD FIST OR NORMAL FIST
        if hand_place < 0.33: # LEFT SCREEN FIST
            return 6 # spin left
        elif hand_place > 0.66: # RIGHT SCREEN FIST
            return 5 # spin right
        else:
            if hand_sign_id == 0: # MIDDLE SCREEN UPWARD FIST
                return 1 # normal forward
            else: # MIDDLE SCREEN NORMAL FIST
                return 7 # fast forward
    elif hand_sign_id == 2: # REVERSE FIST
        return 2 # backward
    elif hand_sign_id == 3: # PALM
        return 0 # stop
    elif hand_sign_id == 4: # REVERSE PALM
        return 8 # do nothing
    elif hand_sign_id == 5: # THUMB LEFT
        return 3 # go left
    elif hand_sign_id == 6: # THUMB RIGHT
        return 4 # go right
    elif hand_sign_id == 7: # OK
        if (previous_action_id != 9): # if change mode first time
            auto_mode = not auto_mode
        return 9 # change mode
    elif hand_sign_id == 8: # INDEX UP
        if (previous_action_id != 10): # if change light first time
            light_mode = not light_mode
        return 10 # change light

if __name__ == '__main__':
    main()

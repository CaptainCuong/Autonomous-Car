#!/usr/bin/env python
# -*- coding: utf-8 -*-
#import csv
import copy
#import argparse
#import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp
import os 
import time
from datetime import datetime 
from serial import Serial 

import serial

s = serial.Serial(port = 'COM3', baudrate=19200, bytesize = 8, timeout = 1)

from helper import *
#from utils import CvFpsCalc
#from model import KeyPointClassifier
#from model import PointHistoryClassifier

cam_width = 960
cam_height = 540
min_detection_confidence  = 0.5
min_tracking_confidence = 0.5

auto_mode = "Manual"
previous_action_id = -1
current_action_id = -1
mpDraw = mp.solutions.drawing_utils
def main():
    # 引数解析 #################################################################
    global previous_action_id
    global current_action_id
    global auto_mode
    global s
    cap_device = 0
    cap_width = cam_width
    cap_height = cam_height

    use_static_image_mode = False
    use_brect = True

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # モデルロード #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    action_list = ["stop", "forward", "backward", "turn left", "turn right", "go left", "go right", "backward left", "backward right", "change mode"]
    keypoint_classifier_labels = ["fist", "reverse fist", "palm", "thumb_left", "thumb_right", "OK", "draw"]
    point_history_classifier_labels = ["swipe_left", "swipe_right", "swipe_up", "swipe_down", "no_motion"]
    finger_action_list = ["auto go left", "auto go right", "turn on light", "turn off light", "no action"]

    # 座標履歴 #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # フィンガージェスチャー履歴 ################################################
    finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = 0

    while True:

        # キー処理(ESC：終了) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        #number, mode = select_mode(key, mode)

        # カメラキャプチャ #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # ミラー表示
        debug_image = copy.deepcopy(image)

        # 検出実施 #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # 外接矩形の計算
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # ランドマークの計算
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # 相対座標・正規化座標への変換
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)
                # 学習データ保存
                #logging_csv(number, mode, pre_processed_landmark_list,
                            #pre_processed_point_history_list)

                # ハンドサイン分類
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 6:  # 指差しサイン
                    point_history.append(landmark_list[8])  # 人差指座標
                else:
                    point_history.append([0, 0])

                # フィンガージェスチャー分類
                finger_gesture_id = -1
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # 直近検出の中で最多のジェスチャーIDを算出
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()

                # 描画
                if (hand_sign_id != 6):
                    previous_action_id = current_action_id
                    current_action_id = getActionAndMode(hand_sign_id, hand_landmarks.landmark[0].x)
                    current_finger_action = "None"
                else: 
                    current_finger_action = finger_action_list[most_common_fg_id[0][0]]


                ## Send to PI
                s.write(str.encode(str(current_action_id)+'.'))
                print(current_action_id)


                ##
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    #keypoint_classifier_labels[hand_sign_id],
                    action_list[current_action_id],
                    #point_history_classifier_labels[most_common_fg_id[0][0]],
                    current_finger_action,
                    auto_mode
                )
        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        #debug_image = draw_info(debug_image, mode, number)
        debug_image = draw_info(debug_image, mode)

        # 画面反映 #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()

def getActionAndMode(hand_sign_id, hand_place):
    global auto_mode
    global previous_action_id
    if hand_sign_id == 0:
        if hand_place < 0.33:
            return 3
        elif hand_place > 0.66:
            return 4
        else:
            return 1
    elif hand_sign_id == 1:
        if hand_place < 0.33:
            return 7
        elif hand_place > 0.66:
            return 8
        else:
            return 2
    elif hand_sign_id == 2:
        return 0
    elif hand_sign_id == 3:
        return 5
    elif hand_sign_id == 4:
        return 6
    else:
        if auto_mode == "Manual" and previous_action_id != 9:
            auto_mode = "Auto"
        elif auto_mode == "Auto" and previous_action_id != 9:
            auto_mode = "Manual"
        return 9



def draw_info(image, mode, number = 0):
    #cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               #1.0, (0, 0, 0), 4, cv.LINE_AA)
    #cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               #1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()

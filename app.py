#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

import wx
from pynput.mouse import Button, Controller

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier




def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cwidth", help='cap width', type=int, default=960)
    parser.add_argument("--cheight", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    parser.add_argument("--inf_middle_hukou",type=float,default=9) #设定最大比值为9
    parser.add_argument("--open_middle_hukou",type=float,default=2) #大于2一般会被判定为张开手的状态
    parser.add_argument("--close_middle_hukou",type=float,default=0.45) #小于0.45认为食指和大拇指并上了

    app=wx.App(False)
    (sw,sh)=wx.GetDisplaySize()

    parser.add_argument("--swidth", help='screen width', type=int, default=sw)
    parser.add_argument("--sheight", help='screen height', type=int, default=sh)

    args = parser.parse_args()

    return args



def main():
    # 参数解析 #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.cwidth
    cap_height = args.cheight

    inf_middle_hukou = args.inf_middle_hukou

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # 准备相机 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # 加载模型 #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=4,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    # 加载标签 ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # fps测量模块 ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # 各种历史记录数组 #################################################################
    porinter_history = deque(maxlen=16) # 食指尖坐标历史记录
    finger_gesture_history = deque(maxlen=16) # 手指手势历史记录
    middle_point_history = deque(maxlen=8) #食指大拇指中间点坐标历史记录
    middle_line_history =  deque(maxlen=8) #食指大拇指中间线长度历史记录
    middle_hukou_history = deque(maxlen=8) #中间线与虎口长度的比值的历史记录

    for i in range(8):
        middle_point_history.append([0.0])
        # middle_line_history.append(9999)
        middle_hukou_history.append(inf_middle_hukou)


    #  ########################################################################
    mode = 0

    while True:
        fps = cvFpsCalc.get()

        # 按键处理（ESC：退出） #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # 获取摄像机内容 #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # 镜像显示
        debug_image = copy.deepcopy(image)

        # 检测手部信息 #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # 可以通过修改image的范围大小限定只识别这个区域内的手 待完善

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # 计算手的外接矩形
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # 计算手的关键点
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # 转换为相对坐标和规范化坐标
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, porinter_history)
                # 保存学习数据
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                # 手势分类
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                
                if hand_sign_id == 2:  # 如果是伸出食指的手势
                    porinter_history.append(landmark_list[8])  # 保存食指坐标
                    middle_point_history.appendleft([0,0])
                    # middle_line_history.appendleft(9999)
                    middle_hukou_history.appendleft(inf_middle_hukou)

                elif hand_sign_id == 3:   # 如果是鼠标的手势
                    middle_point=calc_middle_point(landmark_list[8],landmark_list[4]) #计算中间点
                    middle_line=calc_middle_line(landmark_list[8],landmark_list[4]) #计算中间线长度
                    hukou_line=calc_hukou_line(landmark_list[2],landmark_list[5]) #计算虎口长度

                    debug_image=draw_mouse(debug_image,landmark_list[8],landmark_list[4],middle_point) #画出中间线
                    
                    porinter_history.append([0,0])
                    middle_point_history.appendleft(middle_point)
                    # middle_line_history.appendleft(middle_line)
                    middle_hukou_history.appendleft(middle_line/hukou_line)

                    print(middle_hukou_history)
                    func_mouse(debug_image,middle_point_history,middle_hukou_history)
                    
                else:
                    porinter_history.append([0, 0])
                    middle_point_history.appendleft([0,0])
                    # middle_line_history.appendleft(9999)
                    middle_hukou_history.appendleft(inf_middle_hukou)

                # 手指手势分类
                finger_gesture_id = 6
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (16 * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # 计算最近检测中最多的手势ID
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common() # 根据历史的16个点的数据得到出现最多的手指手势作为置信手势

                # 画出相关数据 fps/外接矩形/关键点/手势信息等
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
        else:
            porinter_history.append([0, 0])

        debug_image = draw_point_history(debug_image, porinter_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        # 显示出来 #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    # 通过输入控制程序
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    #计算外接矩形
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    #计算得到关键点列表
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    # 关键点
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def calc_middle_point(index_landmark_point,thumb_landmark_point):
    # 计算得到食指和中指的中间坐标
    middle_landmark_point = []
    middle_landmark_point.append(int((index_landmark_point[0]+thumb_landmark_point[0])/2))
    middle_landmark_point.append(int((index_landmark_point[1]+thumb_landmark_point[1])/2))
    return middle_landmark_point

def calc_middle_line(index_landmark_point,thumb_landmark_point):
    middle_line = int(
        np.math.sqrt(
        pow(index_landmark_point[0]-thumb_landmark_point[0],2)+
        pow(index_landmark_point[1]-thumb_landmark_point[1],2),
        )
    ) 
    return middle_line

def calc_hukou_line(landmark2,landmark5):
    hukou_line = int(
        np.math.sqrt(
        pow(landmark5[0]-landmark2[0],2)+
        pow(landmark5[1]-landmark2[1],2),
        )
    )
    return hukou_line

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # 转换为相对坐标
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # 转换为一维列表
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # 归一化
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # 转换为相对坐标
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # 转换为一维列表
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_landmarks(image, landmark_point):
    # 画出连接线
    if len(landmark_point) > 0:
        # 大拇指
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # 食指
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # 中指
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # 无名指
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # 小拇指
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # 手掌
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # 画出关键点
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手腕1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手腕2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 拇指：根部
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 拇指：第一关节
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 拇指：指尖
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 食指：根部
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 食指：第二关节
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 食指：第一关节
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 食指：指尖
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：根部
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第二关节
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第一关节
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指尖
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 无名指：根部
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 无名指：第二关节
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 无名指：第一关节
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 无名指：指尖
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小拇指：根部
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小拇指：第二关节
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小拇指：第一关节
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小拇指：指尖
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # 外接矩形
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    #打印手势信息（外接矩形上方的黑色框框）
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    #打印历史点
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)
    return image


def draw_info(image, fps, mode, number):
    #画面左上方打印fps等信息
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

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

def draw_mouse(image,index_point,thumb_point,middle_point):
    # 划出一条连接食指指尖和大拇指指尖的线，并在中点画出点来
    cv.line(image, tuple(index_point), tuple(thumb_point),
                (0,245,255), 6)
    cv.line(image, tuple(index_point), tuple(thumb_point),
                (0,229,238), 2)
    cv.circle(image, (middle_point[0], middle_point[1]), 5, (0,229,238),
                      -1)
    cv.circle(image, (middle_point[0], middle_point[1]), 5, (0,245,255), 1)
    return image

def func_mouse(image,middle_point_history,middle_hukou_history):
    #执行鼠标功能 待完善
    args = get_args()

    inf_middle_hukou = args.inf_middle_hukou
    close_middle_hukou = args.close_middle_hukou

    screen_width = args.swidth
    screen_height = args.sheight


    image_width, image_height = image.shape[1], image.shape[0]

    close_mh_array=[]
    open_mh_array=[]

    for i in range(len(middle_hukou_history)):
        if middle_hukou_history[i]<close_middle_hukou:
            close_mh_array.append(middle_hukou_history[i])
        else :
            open_mh_array.append(middle_hukou_history[i])
    
    print(close_mh_array)

    print(open_mh_array)

    mouse=Controller()
    if max(middle_hukou_history)==inf_middle_hukou:
        pass #存在过往误判 鼠标初始化未完成 不进行操作
    elif len(close_mh_array)==8:
        mouse.click(Button.right)
        print("单击鼠标右键")
    elif middle_hukou_history[0]>close_middle_hukou and middle_hukou_history[1]<close_middle_hukou and len(open_mh_array)>1:
        mouse.click(Button.left)
        print("单击鼠标左键")
    else:
        move_distance_x = (middle_point_history[0][0]-middle_point_history[1][0]) #防颤抖
        move_distance_y = (middle_point_history[0][1]-middle_point_history[1][1])
        if abs(move_distance_x)<3 and abs(move_distance_y)<3:
            pass
        else :
            mx=middle_point_history[0][0]
            my=middle_point_history[0][1]
            mouseLocx=int((mx-0.12*image_width)/(0.6*image_width)*screen_width) #通过按比例缩小使得鼠标操控周围
            mouseLocy=int((my-0.25*image_height)/(0.5*image_height)*screen_height)
            
            if mouseLocx<0:
                mouseLocx=0
                print("鼠标越界")
            elif mouseLocx>=screen_width:
                mouseLocx=screen_width-1
                print("鼠标越界")
            if mouseLocy<0:
                mouseLocy=0
                print("鼠标越界")
            elif mouseLocy>=screen_height:
                mouseLocy=screen_height-1
                print("鼠标越界")
            print("鼠标移动")
            mouseLoc=(mouseLocx,mouseLocy)
            print(mouseLoc)
            mouse.position=mouseLoc
            while mouse.position!=mouseLoc:
                pass

if __name__ == '__main__':
    main()

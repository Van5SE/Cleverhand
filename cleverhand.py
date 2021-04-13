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
from utils import firstLRF #导入必要的类




                
                
def get_args():
    parser = argparse.ArgumentParser()
    #没有后置 前置摄像头为0 有后置 前置摄像头为1
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cwidth", help='cap width', type=int, default=1280)
    parser.add_argument("--cheight", help='cap height', type=int, default=720)

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
    parser.add_argument("--close_middle_hukou",type=float,default=0.5) #小于0.5认为食指和大拇指并上了
    parser.add_argument("--inf_LRF_line",type=int,default=999) #设定左右手的最大间距为999
    parser.add_argument('--inf_hand_angle',type=int,default=999) #设定没有检测到手时的角度为999

    app=wx.App(False)
    (sw,sh)=wx.GetDisplaySize()

    parser.add_argument("--swidth", help='screen width', type=int, default=sw)
    parser.add_argument("--sheight", help='screen height', type=int, default=sh)
    parser.add_argument("--small_angle",type=int,default=60) #设定60度以下为小角度
    parser.add_argument("--big_angle",type=int,default=110) #设定110度以上为大角度


    args = parser.parse_args()

    return args

left_button_flag=False
right_button_flag=False

func_string="Unknow"
func_work_status_flag=False

def main():
    # 参数解析 #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.cwidth
    cap_height = args.cheight

    inf_middle_hukou = args.inf_middle_hukou
    inf_LRF_line=args.inf_LRF_line

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    global func_work_status_flag
    global func_string
    use_brect = True

    # 准备相机 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # 加载模型 #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    # 加载模型 ###########################################################
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
    # cvFpsCalc = CvFpsCalc(buffer_len=10)

    # 各种历史记录数组 #################################################################
    pointer_history = deque(maxlen=16) # 食指尖坐标历史记录
    finger_gesture_history = deque(maxlen=16) # 手指手势历史记录
    middle_point_history = deque(maxlen=8) #食指大拇指中间点坐标历史记录
    middle_hukou_history = deque(maxlen=8) #中间线与虎口长度的比值的历史记录
    mouse_point_history = deque(maxlen=8) #指示鼠标、抓取等关键点的坐标点位置
    LRF_line_history = deque(maxlen=8) #存储左右手指间距离的历史记录
    LRF_point_history = deque(maxlen=2) #存储左右手指尖坐标的历史记录
    hand_angle_history = deque(maxlen=16) #存储手的朝向角度的历史记录
    firstLRFdata = firstLRF() #存储下第一个左右手指数据


    def append_other_deque(deque_1=None,deque_2=None,deque_3=None,firstLRF_1=None):
    # 将没用到的数据结构添加一位空位
        nonlocal firstLRFdata
        if(pointer_history!=deque_1 and pointer_history!=deque_2 and pointer_history!=deque_3):
            pointer_history.append([0,0])
        if(middle_point_history!=deque_1 and middle_point_history!=deque_2 and middle_point_history!=deque_3):
            middle_point_history.appendleft([0,0])
        if(middle_hukou_history!=deque_1 and middle_hukou_history!=deque_2 and middle_hukou_history!=deque_3):
            middle_hukou_history.appendleft(inf_middle_hukou)
        if(mouse_point_history!=deque_1 and mouse_point_history!=deque_2 and mouse_point_history!=deque_3):
            mouse_point_history.appendleft([0,0])
        if(LRF_line_history!=deque_1 and LRF_line_history!=deque_2 and LRF_line_history!=deque_3):
            LRF_line_history.appendleft(inf_LRF_line)
        if(LRF_point_history!=deque_1 and LRF_point_history!=deque_2 and LRF_point_history!=deque_3):
            LRF_point_history.appendleft([0,0])
        if(firstLRFdata!=firstLRF_1 and firstLRFdata.leftF!=None): #TODO 
            firstLRFdata=firstLRF()
        if(hand_angle_history!=deque_1 and hand_angle_history!=deque_2 and hand_angle_history!=deque_3):
            hand_angle_history.appendleft(0)
        


    for i in range(8):
        pointer_history.append([0,0])
        pointer_history.append([0,0])
        middle_point_history.append([0,0])
        middle_hukou_history.append(inf_middle_hukou)
        mouse_point_history.append([0,0])
        LRF_line_history.append(0)
        LRF_point_history.append([0,0])
        hand_angle_history.append(0)
        hand_angle_history.append(0)


    #  ########################################################################
    logMode = 0

    while True:
        # fps = cvFpsCalc.get()

        # 按键处理（ESC：退出） #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        logNumber, logMode = select_mode(key, logMode)

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

        func_work_status_flag=False

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
                    debug_image, pointer_history)
                # 保存学习数据
                logging_csv(logNumber, logMode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                # 手势分类
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 0 and len(results.multi_handedness)==2 : # 张开手掌的两只手手势
                    
                    if(handedness.classification[0].label[0:]=="Right"):
                        LRF_point_history[1]=landmark_list[8]
                    elif(handedness.classification[0].label[0:]=="Left"):
                        LRF_point_history[0]=landmark_list[8] # 0存储左手位置 1存储右手位置
                    
                    if(LRF_point_history[1]!=[0,0] and LRF_point_history[0]!=[0,0]): #LRF=Left Right Finger
                        LRF_line=int(calc_middle_line(LRF_point_history[0],LRF_point_history[1])/cap_width*500) # 将长度归一化
                        if(LRF_line==0):
                            LRF_line=1
                        LRF_line_history.appendleft(LRF_line) #计算得到LRF_line 这个history数组可能不需要
                        #print(LRF_line)
                        debug_image=draw_mouse(debug_image,LRF_point_history[0],LRF_point_history[1])

                        #print(LRF_point_history[0],LRF_point_history[1])
                        LRF_angle=calc_angle(LRF_point_history[0],LRF_point_history[1]) #计算得到LRFangle
                        #print(LRFangle)

                        if(firstLRFdata.leftF==None and max(LRF_point_history)!=inf_LRF_line): #TODO
                            firstLRFdata=firstLRF(LRF_point_history[0],LRF_point_history[1],LRF_line,LRF_angle)
                        elif (max(LRF_point_history)!=inf_LRF_line): #判断不要快速跳动
                            func_switch_two_open(firstLRFdata,LRF_line,LRF_angle)

                    #print(firstLRFdata.leftF)
                    #print(LRF_point_history)
                    #print(LRF_line_history)
                    
                    append_other_deque(pointer_history,LRF_line_history,LRF_point_history,firstLRF_1=firstLRFdata)

                    #debug_image = draw_pointer(debug_image, pointer_history)

                elif hand_sign_id == 0 and len(results.multi_handedness)==1 : # 张开手掌的一只手手势
                    hand_angle=calc_hand_angle(landmark_list[0],landmark_list[9])
                    #print(hand_angle)
                    hand_angle_history.appendleft(hand_angle)

                    func_slide(hand_angle_history)
                    
                    append_other_deque(hand_angle_history)
                    
                elif hand_sign_id == 1:   # 如果是鼠标的手势
                    middle_point=calc_middle_point(landmark_list[8],landmark_list[4]) #计算中间点
                    middle_line=calc_middle_line(landmark_list[8],landmark_list[4]) #计算中间线长度
                    hukou_line=calc_hukou_line(landmark_list[2],landmark_list[5]) #计算虎口长度

                    middle_point_history.appendleft(middle_point)
                    middle_hukou_history.appendleft(middle_line/hukou_line)
                    mouse_point_history.appendleft(landmark_list[12])
                    append_other_deque(middle_point_history,middle_hukou_history,mouse_point_history) #记录数据

                    debug_image=draw_mouse(debug_image,landmark_list[8],landmark_list[4],mouse_point_history[0]) #画出中间线
                    func_mouse(debug_image,mouse_point_history,middle_hukou_history)

                elif hand_sign_id == 2:   # 如果是抓住的手势
                    middle_point=calc_middle_point(landmark_list[8],landmark_list[4]) #计算中间点
                    middle_line=calc_middle_line(landmark_list[8],landmark_list[4]) #计算中间线长度
                    hukou_line=calc_hukou_line(landmark_list[2],landmark_list[5]) #计算虎口长度

                    middle_point_history.appendleft(middle_point)
                    middle_hukou_history.appendleft(middle_line/hukou_line)
                    mouse_point_history.appendleft(middle_point) #暂时还是用两连线中点作为鼠标位置 TODO
                    append_other_deque(middle_point_history,middle_hukou_history,mouse_point_history) #记录数据

                    debug_image=draw_mouse(debug_image,landmark_list[8],landmark_list[4],mouse_point_history[0]) #画出中间线
                    func_grab(debug_image,mouse_point_history,middle_hukou_history)

                elif hand_sign_id == 4:  # 如果是1的手势
                    pointer_history.append(landmark_list[8])  # 保存食指坐标
                    append_other_deque(pointer_history)

                    debug_image = draw_pointer(debug_image, pointer_history)

                else:  
                    append_other_deque()

                # 手指手势分类
                finger_gesture_id = 6
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (16 * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # 计算最近检测中最多的手势ID
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()[0][0] # 根据历史的16个点的数据得到出现最多的手指手势作为置信手势

                # 画出相关数据 fps/外接矩形/关键点/手势信息等
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id],
                )
            debug_image = draw_image(debug_image)
        else:
            append_other_deque()
            debug_image = draw_image(debug_image)
            pass
 
        debug_image = draw_info(debug_image, func_string,func_work_status_flag, logMode, logNumber)

        # 显示出来 #############################################################
        cv.imshow('CleverHand', debug_image)

    cap.release()
    cv.destroyAllWindows()



def select_mode(key, logMode):
    # 通过输入控制程序
    logNumber = -1
    if 48 <= key <= 57:  # 0 ~ 9
        logNumber = key - 48
    if key == 110:  # n
        logMode = 0
    if key == 107:  # k
        logMode = 1
    if key == 104:  # h
        logMode = 2
    return logNumber, logMode

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
        # 由归一化坐标扩展到真实坐标
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

def calc_angle(lp1, rp1,lp2=[0,0],rp2=[1,0]): #lp1 = leftpoint1 rp1 = rightpoint1 可以传入四个点 1和2两组线计算其夹角
    dx1 = rp1[0] - lp1[0]
    dy1 = rp1[1] - lp1[1]
    dx2 = rp2[0] - lp2[0]
    dy2 = rp2[1] - lp2[1]
    angle1 = np.math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180/np.math.pi)
    #print(angle1)
    angle2 = np.math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180/np.math.pi)
    #print(angle2)
    if angle1*angle2 >= 0:
        included_angle = angle1-angle2
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
        else:
            included_angle=-included_angle
    return included_angle

def calc_hand_angle(lm0,lm9,lp2=[0,0],rp2=[0,1]): #lm0=landmark0 lm9=landmark9
    dx1 = lm9[0] - lm0[0]
    dy1 = lm9[1] - lm0[1]
    dx2 = rp2[0] - lp2[0]
    dy2 = rp2[1] - lp2[1]
    angle1 = np.math.atan2(-dy1, dx1)
    angle1 = int(angle1 * 180/np.math.pi)
    #print(angle1)
    angle2 = np.math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180/np.math.pi)
    #print(angle2)
    if angle1*angle2 >= 0:
        included_angle = angle1-angle2
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
        else:
            included_angle=-included_angle
    return included_angle


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
    # 归一化处理
        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # 转换为一维列表
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
# 将手掌坐标记入csv中
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoints/keypoint'+str(number)+'.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_historys/point_history'+str(number)+'.csv'
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
# 外接矩形
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,finger_gesture_text):
#打印手势信息（动态手势信息和黑框内的信息）
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    image_width, image_height = image.shape[1], image.shape[0]
    #image=image[int(0.02*image_height):int(0.98*image_height),int(0.02*image_width):int(0.98*image_width)]

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (int(0.02*image_width)+10, int(0.02*image_height)+60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (int(0.02*image_width)+10, int(0.02*image_height)+60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_pointer(image, point_history):
#打印历史点
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)
    return image

def draw_info(image, func_string,func_work_status_flag, logMode, number):
#画面左上方打印当前动作等信息
    if(func_work_status_flag):
        cv.putText(image, "Current action:" + func_string, (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                1.0, (0,0,0), 4, cv.LINE_AA)
        cv.putText(image, "Current action:" + func_string, (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                1.0, (255,255,255), 2, cv.LINE_AA)
    else:
        cv.putText(image, "Last action:" + func_string, (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                1.0, (0,0,0), 4, cv.LINE_AA)
        cv.putText(image, "Last action:" + func_string, (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= logMode <= 2:
        cv.putText(image, "LOGMODE:" + mode_string[logMode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image

def draw_image(image):
    image_width, image_height = image.shape[1], image.shape[0]
    image=image[int(0.02*image_height):int(0.98*image_height),int(0.02*image_width):int(0.98*image_width)]
    return image

def draw_mouse(image,point_1,point_2,middle_point=None):
# 划出一条连接两个坐标点的线，并画出其中点
    cv.line(image, tuple(point_1), tuple(point_2),
                (0,245,255), 6)
    cv.line(image, tuple(point_1), tuple(point_2),
                (0,229,238), 2)
    if(middle_point==None):
        pass
    else:
        cv.circle(image, (middle_point[0], middle_point[1]), 10, (0,229,238),
                        -1)
        cv.circle(image, (middle_point[0], middle_point[1]), 10, (0,245,255), 1)
    return image

def func_mouse(image,mouse_point_history,middle_hukou_history):
#执行鼠标功能 TODO
    args = get_args()

    inf_middle_hukou = args.inf_middle_hukou
    close_middle_hukou = args.close_middle_hukou

    screen_width = args.swidth
    screen_height = args.sheight

    global right_button_flag
    global left_button_flag
    global func_work_status_flag
    global func_string
    func_work_status_flag=True

    image_width, image_height = image.shape[1], image.shape[0]

    close_mh_array=[]
    open_mh_array=[]

    for i in range(len(middle_hukou_history)):
        if middle_hukou_history[i]<close_middle_hukou:
            close_mh_array.append(middle_hukou_history[i])
        else :
            open_mh_array.append(middle_hukou_history[i])
    
    mouse=Controller()
    if max(middle_hukou_history)==inf_middle_hukou:
        pass #存在过往误判 鼠标初始化未完成 不进行操作
    elif len(close_mh_array)>6:
        if(right_button_flag==False):
            mouse.press(Button.right)
            right_button_flag=True
            print("按下Right")
        else:
            print("保持Right按下")
    elif middle_hukou_history[0]>close_middle_hukou and middle_hukou_history[1]<close_middle_hukou and len(open_mh_array)>2:
        mouse.click(Button.left,2)
        print("双击Left")
        func_work_status_flag=True
        func_string='Double Click Left'
    else:
        if(right_button_flag==True):
            mouse.release(Button.right)
            right_button_flag=False
            print("松开Right")
            func_work_status_flag=True
            func_string='Singal Click Right'
        move_distance_x = (mouse_point_history[0][0]-mouse_point_history[1][0]) #防颤抖
        move_distance_y = (mouse_point_history[0][1]-mouse_point_history[1][1])
        if abs(move_distance_x)<3 and abs(move_distance_y)<3:
            pass
        else :
            mx=mouse_point_history[0][0]
            my=mouse_point_history[0][1]
            mouseLocx=int((mx-0.18*image_width)/(0.6*image_width)*screen_width) #通过按比例缩小使得鼠标操控周围
            mouseLocy=int((my-0.25*image_height)/(0.5*image_height)*screen_height)
            
            if mouseLocx<0:
                mouseLocx=0
                # print("鼠标越界")
            elif mouseLocx>=screen_width:
                mouseLocx=screen_width-1
              # print("鼠标越界")
            if mouseLocy<0:
                mouseLocy=0
                # print("鼠标越界")
            elif mouseLocy>=screen_height:
                mouseLocy=screen_height-1
                # print("鼠标越界")
            # print("鼠标移动")
            mouseLoc=(mouseLocx,mouseLocy)
            # print(mouseLoc)
            mouse.position=mouseLoc
            while mouse.position!=mouseLoc:
                pass

def func_grab(image,mouse_point_history,middle_hukou_history):
#执行抓取功能 TODO
    args = get_args()

    inf_middle_hukou = args.inf_middle_hukou
    close_middle_hukou = args.close_middle_hukou

    screen_width = args.swidth
    screen_height = args.sheight

    global left_button_flag
    global func_work_status_flag
    global func_string

    image_width, image_height = image.shape[1], image.shape[0]

    close_mh_array=[]
    open_mh_array=[]

    for i in range(len(middle_hukou_history)):
        if middle_hukou_history[i]<close_middle_hukou:
            close_mh_array.append(middle_hukou_history[i])
        else :
            open_mh_array.append(middle_hukou_history[i])
    
    mouse=Controller()
    if max(middle_hukou_history)==inf_middle_hukou:
        pass #存在过往误判 鼠标初始化未完成 不进行操作
    elif len(close_mh_array)>=4:
        func_work_status_flag=True
        func_string='Press Left And Move'
        if(left_button_flag==False): #鼠标左键没按下
            mouse.press(Button.left) # 按下鼠标左键进行拖动
            left_button_flag=True
            print("按下Left")
        else:
            print("保持Left按下")
        move_distance_x = (mouse_point_history[0][0]-mouse_point_history[1][0]) #防颤抖
        move_distance_y = (mouse_point_history[0][1]-mouse_point_history[1][1])
        if abs(move_distance_x)<3 and abs(move_distance_y)<3:
            pass
        else :
            mx=mouse_point_history[0][0]
            my=mouse_point_history[0][1]
            mouseLocx=int((mx-0.05*image_width)/(0.6*image_width)*screen_width) #通过按比例缩小使得鼠标操控周围
            mouseLocy=int((my-0.25*image_height)/(0.5*image_height)*screen_height)
            
            if mouseLocx<0:
                mouseLocx=0
                # print("鼠标越界")
            elif mouseLocx>=screen_width:
                mouseLocx=screen_width-1
                # print("鼠标越界")
            if mouseLocy<0:
                mouseLocy=0
                # print("鼠标越界")
            elif mouseLocy>=screen_height:
                mouseLocy=screen_height-1
                # print("鼠标越界")
            # print("鼠标移动")
            mouseLoc=(mouseLocx,mouseLocy)
            # print(mouseLoc)
            mouse.position=mouseLoc
            while mouse.position!=mouseLoc:
                pass
    else:
        func_work_status_flag=True
        func_string='Release Left And Move'
        if(left_button_flag==True):
            mouse.release(Button.left) # 松开鼠标左键进行拖动
            print("松开Left")
            left_button_flag=False
        else:
            print("保持Left松开")
        move_distance_x = (mouse_point_history[0][0]-mouse_point_history[1][0]) #防颤抖
        move_distance_y = (mouse_point_history[0][1]-mouse_point_history[1][1])
        if abs(move_distance_x)<3 and abs(move_distance_y)<3:
            pass
        else :
            mx=mouse_point_history[0][0]
            my=mouse_point_history[0][1]
            mouseLocx=int((mx-0.12*image_width)/(0.6*image_width)*screen_width) #通过按比例缩小使得鼠标操控周围
            mouseLocy=int((my-0.25*image_height)/(0.5*image_height)*screen_height)
            
            if mouseLocx<0:
                mouseLocx=0
                # print("鼠标越界")
            elif mouseLocx>=screen_width:
                mouseLocx=screen_width-1
                # print("鼠标越界")
            if mouseLocy<0:
                mouseLocy=0
                # print("鼠标越界")
            elif mouseLocy>=screen_height:
                mouseLocy=screen_height-1
                # print("鼠标越界")
            # print("鼠标移动")
            mouseLoc=(mouseLocx,mouseLocy)
            # print(mouseLoc)
            mouse.position=mouseLoc
            while mouse.position!=mouseLoc:
                pass

def func_slide(hand_angle_history):
    # 执行上一张下一张功能
    args=get_args()
    global func_work_status_flag
    global func_string
    big_angle=args.big_angle
    small_angle=args.small_angle
    pos_hand_angle_list=[]
    neg_hand_angle_list=[]
    big_hand_angle_list=[]
    small_hand_angle_list=[]
    for i in range(len(hand_angle_history)):
        if (hand_angle_history[i]>0):
            pos_hand_angle_list.append(hand_angle_history[i])
        elif (hand_angle_history[i]<0):
            neg_hand_angle_list.append(hand_angle_history[i])
        if (abs(hand_angle_history[i])>big_angle):
            big_hand_angle_list.append(hand_angle_history[i])
        elif (abs(hand_angle_history[i])<small_angle and hand_angle_history[i]!=0):
            small_hand_angle_list.append(hand_angle_history[i])


    print(hand_angle_history)
    print(len(big_hand_angle_list))
    print(len(small_hand_angle_list))

    if deque.count(hand_angle_history,0)>=6:
        pass
    elif hand_angle_history[0]<0 and hand_angle_history[-1]>0 and len(neg_hand_angle_list)==4 and len(pos_hand_angle_list)>=6 and hand_angle_history[-1]-hand_angle_history[0]>40:
        print("下一张")
        func_work_status_flag=True
        func_string="Next One"
    elif hand_angle_history[0]>0 and hand_angle_history[-1]<0 and len(pos_hand_angle_list)==4 and len(neg_hand_angle_list)>=6 and hand_angle_history[0]-hand_angle_history[-1]>40:
        print("上一张")
        func_work_status_flag=True
        func_string="Last One"
    elif abs(hand_angle_history[0])<small_angle and abs(hand_angle_history[-1])>big_angle and len(small_hand_angle_list)==4 and len(big_hand_angle_list)>=4:
        print("向上")
        func_work_status_flag=True
        func_string="Up"
    elif abs(hand_angle_history[0])>big_angle and abs(hand_angle_history[-1])<small_angle and len(big_hand_angle_list)==4 and len(small_hand_angle_list)>=4:
        print("向下")
        func_work_status_flag=True
        func_string="Down"
    

    

def func_switch_two_open(firstLRFdata,LRF_line,LRF_angle):
#对两个张开手势的运动结果的功能切换
    global func_work_status_flag
    global func_string
    func_work_status_flag=True
    LRF_line_sub=LRF_line-firstLRFdata.LRF_line
    LRF_angle_sub=LRF_angle-firstLRFdata.LRF_angle
    #print(LRF_line,firstLRFdata.LRF_line,LRF_angle,firstLRFdata.LRF_angle)
    #print(LRF_line_sub,LRF_angle_sub)
    if(abs(LRF_line_sub)>0.3*firstLRFdata.LRF_line):
        if(LRF_line_sub>0):
            print("放大：",abs(LRF_line_sub)/firstLRFdata.LRF_line*0.8)
            func_string=("Zoom on:"+str(abs(LRF_line_sub)/firstLRFdata.LRF_line*0.8))
        elif(LRF_line_sub<0):
            print("缩小：",abs(LRF_line_sub)/firstLRFdata.LRF_line*1.2)
            func_string=("Zoom out:"+str(abs(LRF_line_sub)/firstLRFdata.LRF_line*1.2))
    elif(abs(LRF_angle_sub)>10): 
        if(LRF_angle_sub>0):
            print("顺时针旋转",abs(LRF_angle_sub*2)-20,"度")
            func_string=("Clockwise:"+str(abs(LRF_angle_sub*2)-20)+"degree")
        elif(LRF_angle_sub<0):
            print("逆时针旋转",abs(LRF_angle_sub*2)-20,"度")
            func_string=("C-Clockwise:"+str(abs(LRF_angle_sub*2)-20)+"degree")
    else:
        print("不动")
        func_string=("Stay")


if __name__ == '__main__':
    main()


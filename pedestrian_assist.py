import pathlib
import os
import sys
import cv2
import torch
import numpy as np
import math
import timeit
import serial
import time
from sklearn import linear_model
from PIL import Image

# ─────────────────────────────────────────
# Windows 환경 경로 호환성 처리
# ─────────────────────────────────────────
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# ─────────────────────────────────────────
# 설정 (환경변수 or 기본값)
# ─────────────────────────────────────────
DARKNET_PATH  = os.getenv("DARKNET_PATH",  "darknet")
MODEL_CFG     = os.getenv("MODEL_CFG",     "config/custom.cfg")
MODEL_DATA    = os.getenv("MODEL_DATA",    "config/trDatas.data")
MODEL_WEIGHTS = os.getenv("MODEL_WEIGHTS", "weights/yolov4-tiny-custom_final.weights")
SERIAL_PORT   = os.getenv("SERIAL_PORT",   "/dev/ttyACM0")
SERIAL_BAUD   = int(os.getenv("SERIAL_BAUD", "115200"))
CAM_INDEX_1   = int(os.getenv("CAM_INDEX_1", "0"))  # 횡단보도 카메라
CAM_INDEX_2   = int(os.getenv("CAM_INDEX_2", "1"))  # 신호등 카메라

# ─────────────────────────────────────────
# Darknet 로드
# ─────────────────────────────────────────
sys.path.insert(0, DARKNET_PATH)
import darknet

network, class_names, class_colors = darknet.load_network(
    MODEL_CFG,
    MODEL_DATA,
    MODEL_WEIGHTS
)

# ─────────────────────────────────────────
# 함수 정의
# ─────────────────────────────────────────

def lineCalc(vx, vy, x0, y0):
    """두 점으로 직선의 기울기(m)와 절편(b) 계산"""
    scale = 10
    x1 = x0 + scale * vx
    y1 = y0 + scale * vy
    m = (y1 - y0) / (x1 - x0)
    b = y1 - m * x1
    return m, b


def lineIntersect(m1, b1, m2, b2):
    """두 직선의 교점(소실점) 계산"""
    a_1, b_1, c_1 = -m1, 1, b1
    a_2, b_2, c_2 = -m2, 1, b2
    d  = a_1 * b_2 - a_2 * b_1
    dx = c_1 * b_2 - c_2 * b_1
    dy = a_1 * c_2 - a_2 * c_1
    return dx / d, dy / d


def process(im):
    """
    횡단보도 라인 인식 및 방향 계산
    - 흰색 라인 마스킹 → 윤곽선 추출 → RANSAC 직선 피팅 → 소실점 계산
    Returns: (처리된 프레임, Dx, Dy)
    """
    x = W
    y = H
    radius   = 250
    bw_width = 170

    bxbyLeftArray  = []
    bxbyRightArray = []
    boundedLeft    = []
    boundedRight   = []

    # 흰색 영역 마스킹
    lower = np.array([170, 170, 170])
    upper = np.array([255, 255, 255])
    mask  = cv2.inRange(im, lower, upper)

    # 침식으로 노이즈 제거
    erodeSize      = int(y / 30)
    erodeStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (erodeSize, 1))
    erode          = cv2.erode(mask, erodeStructure, (-1, -1))

    contours, _ = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for i in contours:
        bx, by, bw, bh = cv2.boundingRect(i)
        if bw > bw_width:
            cv2.line(im, (bx, by), (bx + bw, by), (0, 255, 0), 2)
            bxbyLeftArray.append([bx, by])
            bxbyRightArray.append([bx + bw, by])
            cv2.circle(im, (int(bx), int(by)), 5, (0, 250, 250), 2)
            cv2.circle(im, (int(bx + bw), int(by)), 5, (250, 250, 0), 2)

    medianL = np.median(bxbyLeftArray,  axis=0)
    medianR = np.median(bxbyRightArray, axis=0)

    bxbyLeftArray  = np.asarray(bxbyLeftArray)
    bxbyRightArray = np.asarray(bxbyRightArray)

    # 중앙값 기준 반경 내 포인트만 사용
    for i in bxbyLeftArray:
        if ((medianL[0] - i[0])**2 + (medianL[1] - i[1])**2) < radius**2:
            boundedLeft.append(i)
    for i in bxbyRightArray:
        if ((medianR[0] - i[0])**2 + (medianR[1] - i[1])**2) < radius**2:
            boundedRight.append(i)

    boundedLeft  = np.asarray(boundedLeft)
    boundedRight = np.asarray(boundedRight)

    bxLeft  = np.asarray(boundedLeft[:, 0])
    byLeft  = np.asarray(boundedLeft[:, 1])
    bxRight = np.asarray(boundedRight[:, 0])
    byRight = np.asarray(boundedRight[:, 1])

    # RANSAC 직선 피팅
    model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
    model_ransac.fit(np.array([bxLeft]).T,  byLeft)
    inlier_maskL = model_ransac.inlier_mask_
    model_ransac.fit(np.array([bxRight]).T, byRight)
    inlier_maskR = model_ransac.inlier_mask_

    for element in boundedRight[inlier_maskR]:
        cv2.circle(im, (element[0], element[1]), 10, (250, 250, 250), 2)
    for element in boundedLeft[inlier_maskL]:
        cv2.circle(im, (element[0], element[1]), 10, (100, 100, 250), 2)

    vx,   vy,   x0,   y0   = cv2.fitLine(boundedLeft[inlier_maskL],  cv2.DIST_L2, 0, 0.01, 0.01)
    vx_R, vy_R, x0_R, y0_R = cv2.fitLine(boundedRight[inlier_maskR], cv2.DIST_L2, 0, 0.01, 0.01)

    m_L, b_L = lineCalc(vx,   vy,   x0,   y0)
    m_R, b_R = lineCalc(vx_R, vy_R, x0_R, y0_R)

    intersectionX, intersectionY = lineIntersect(m_R, b_R, m_L, b_L)

    m = radius * 10
    if intersectionY < H / 2:
        cv2.circle(im, (int(intersectionX), int(intersectionY)), 10, (0, 0, 255), 15)
        cv2.line(im,
                 (int(x0   - m * vx),   int(y0   - m * vy)),
                 (int(x0   + m * vx),   int(y0   + m * vy)),   (255, 0, 0), 3)
        cv2.line(im,
                 (int(x0_R - m * vx_R), int(y0_R - m * vy_R)),
                 (int(x0_R + m * vx_R), int(y0_R + m * vy_R)), (255, 0, 0), 3)

    POVx = W / 2
    POVy = H / 2
    Dx = -int(intersectionX - POVx)
    Dy = -int(intersectionY - POVy)

    return im, Dx, Dy


# ─────────────────────────────────────────
# 메인 루프
# ─────────────────────────────────────────
arduino = serial.Serial(SERIAL_PORT, SERIAL_BAUD)

cap1 = cv2.VideoCapture(CAM_INDEX_1)  # 횡단보도 인식 카메라
cap2 = cv2.VideoCapture(CAM_INDEX_2)  # 신호등 인식 카메라

W = cap1.get(3)
H = cap1.get(4)
ratio = H / W
W = 800
H = int(W * ratio)

Dx_list    = []
Dy_list    = []
DxAve      = 0
DyAve      = 0
Dxold      = 0
Dyold      = 0
i          = 0
state      = ""
traffic_light = 0

print("실행 중... 종료하려면 'q'를 누르세요.")

while cap1.isOpened() and cap2.isOpened():
    ret,  frame  = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret or not ret2:
        print("프레임을 가져올 수 없습니다.")
        break

    # ── 횡단보도 카메라 전처리 ──
    frame_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_img = Image.fromarray(frame_rgb).resize((W, H))
    img = cv2.cvtColor(np.array(resized_img), cv2.COLOR_RGB2BGR)
    cv2.circle(img, (int(W / 2), int(H / 2)), 5, (0, 0, 255), 8)

    # ── 신호등 인식 (YOLOv4) ──
    darknet_image = darknet.make_image(
        darknet.network_width(network),
        darknet.network_height(network), 3
    )
    frame_rgb2    = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(
        frame_rgb2,
        (darknet.network_width(network), darknet.network_height(network)),
        interpolation=cv2.INTER_LINEAR
    )
    darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
    detections  = darknet.detect_image(network, class_names, darknet_image)
    yolo_result = darknet.print_detections(detections)

    traffic_light = 1 if yolo_result == "Green-Pedestrian-signals" else 0
    print(f"신호등 상태: {'초록불 🟢' if traffic_light else '빨간불 🔴'}")

    # ── 횡단보도 라인 인식 및 방향 판단 ──
    try:
        processedFrame, dx, dy = process(img)

        if i < 6:
            Dx_list.append(dx)
            Dy_list.append(dy)
            i += 1
        else:
            DxAve = sum(Dx_list) / len(Dx_list)
            DyAve = sum(Dy_list) / len(Dy_list)
            Dx_list.clear()
            Dy_list.clear()
            i = 0

        if DyAve > 30 and abs(DxAve) < 300 and traffic_light == 1:
            if ((DxAve - Dxold)**2 + (DyAve - Dyold)**2) < 150**2:
                cv2.line(img,
                         (int(W / 2), int(H / 2)),
                         (int(W / 2) + int(DxAve), int(H / 2) + int(DyAve)),
                         (0, 0, 255), 7)

                # 방향 판단 및 아두이노 전송
                if abs(DxAve) < 80 and DyAve > 100 and abs(Dxold - DxAve) < 20:
                    state = 'Straight'
                    data  = "S"
                elif DxAve > 80 and DyAve > 100 and abs(Dxold - DxAve) < 20:
                    state = 'Right'
                    data  = "R"
                elif DxAve < 80 and DyAve > 100 and abs(Dxold - DxAve) < 20:
                    state = 'Left'
                    data  = "L"
                else:
                    data = "0"

                print(f"방향: {state}")
                arduino.write(data.encode())

                color = (0, 0, 0) if state == 'Straight' else (0, 0, 255)
                cv2.putText(img, state, (50, 50),
                            cv2.FONT_HERSHEY_PLAIN, 3, color, 2, cv2.LINE_AA)

        Dxold = DxAve
        Dyold = DyAve

        cv2.imshow("보행 보조 시스템", processedFrame)

    except Exception as e:
        print(f"처리 중 오류: {e}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
arduino.close()

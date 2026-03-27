# 🦯 시각장애인 횡단보도 보행 보조 로봇 — 영상처리 모듈

> YOLOv4 신호등 인식 + OpenCV 횡단보도 라인 인식을 결합하여  
> 시각장애인이 안전하게 횡단보도를 건널 수 있도록 방향을 안내하는 시스템

---

## 📌 프로젝트 개요

시각장애인의 횡단보도 보행을 보조하는 로봇 시스템의 **영상처리 담당 모듈**입니다.  
두 개의 카메라를 활용하여 신호등 상태와 횡단보도 방향을 실시간으로 인식하고,  
그 결과를 MCU(아두이노)로 전송하면 로봇이 방향을 제어합니다.

> 팀 프로젝트 — 본 레포지토리는 영상처리 담당 역할의 코드입니다.  
> MCU 제어 로직은 별도 팀원이 담당하였습니다.

---

## 🎯 주요 기능

- **신호등 인식**: YOLOv4-tiny 모델로 보행자 신호(초록/빨간) 실시간 감지
- **횡단보도 라인 인식**: OpenCV + RANSAC으로 횡단보도 방향 계산
- **방향 판단**: 소실점(Vanishing Point) 기반으로 Straight / Left / Right 결정
- **아두이노 연동**: 판단 결과를 시리얼 통신으로 MCU에 전송

---

## 🛠 기술 스택

| 분야 | 기술 |
|------|------|
| 신호등 인식 | YOLOv4-tiny (Darknet), Jetson Nano |
| 횡단보도 인식 | OpenCV, RANSAC (scikit-learn) |
| 언어 | Python |
| 하드웨어 | Jetson Nano, Arduino, 듀얼 카메라 |
| 데이터 라벨링 | 수작업 라벨링 (XML 포맷) |

---

## 🔄 시스템 파이프라인

```
[카메라 1] 횡단보도 영상
    ↓
흰색 라인 마스킹 → 윤곽선 추출 → RANSAC 직선 피팅
    ↓
소실점 계산 → 방향 판단 (Straight / Left / Right)
    ↓
                         ┌─────────────────────────┐
[카메라 2] 신호등 영상 → YOLOv4-tiny 추론          │
                         └─────────────────────────┘
    ↓
초록불 + 방향 데이터 → 아두이노 시리얼 전송 (S / L / R)
    ↓
MCU 제어 → 로봇 이동
```

---

## ⚙️ 개발 환경 및 제약 해결

**Jetson Nano 환경 제약 극복**

YOLOv5는 Jetson Nano에서 동작하지 않아 YOLOv4-tiny로 전환하였습니다.  
또한 Roboflow가 YOLOv4 포맷을 지원하지 않아 라벨링을 전부 수작업으로 진행하였습니다.

| 항목 | 내용 |
|------|------|
| 학습 환경 | Jetson Nano |
| 데이터 수집 | 직접 촬영 |
| 라벨링 | 수작업 XML 포맷 |
| 클래스 | Green-Pedestrian-signals, Red-Pedestrian-signals |

---

## 📊 성능

| 항목 | 내용 |
|------|------|
| 모델 | YOLOv4-tiny |
| 동작 환경 | Jetson Nano 실시간 추론 |
| 신호등 인식 클래스 | 초록불 / 빨간불 |

---

## 🚀 실행 방법

```bash
# 저장소 클론
git clone https://github.com/your-username/pedestrian-assist-vision.git
cd pedestrian-assist-vision

# 패키지 설치
pip install -r requirements.txt

# 환경변수 설정 (선택)
export DARKNET_PATH=darknet
export MODEL_CFG=config/custom.cfg
export MODEL_DATA=config/trDatas.data
export MODEL_WEIGHTS=weights/yolov4-tiny-custom_final.weights
export SERIAL_PORT=/dev/ttyACM0

# 실행
python pedestrian_assist.py
```

---

## 📸 결과 예시

<!-- 시연 사진 또는 GIF 추가 예정 -->
> 횡단보도 인식 및 방향 판단 시연 영상 추가 예정

---

## 💡 담당 역할 및 배운 점

**담당 역할**
- YOLOv4-tiny 신호등 인식 모델 — 데이터 수집, 수작업 라벨링, 학습 전 과정
- OpenCV 횡단보도 라인 인식 코드 수정 및 통합
- 두 인식 결과를 결합하여 MCU 전송 인터페이스 설계

**배운 점**
- 임베디드 환경(Jetson Nano)의 제약 안에서 적합한 모델을 선택하는 경험
- 실제 야외 환경 데이터의 다양성과 라벨링 품질의 중요성 체감
- 영상처리 결과물을 하드웨어 제어 신호로 변환하는 시스템 설계 경험

---

## 👤 개발자

- GitHub: [@your-username](https://github.com/yjs0601)

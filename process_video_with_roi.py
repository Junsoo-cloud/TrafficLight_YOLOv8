import cv2
from ultralytics import YOLO
import serial
import os
import time

# YOLOv8 모델 로드
model = YOLO("yolov8n.pt")  # Pre-trained YOLOv8 모델 사용

# 시리얼 포트 설정 (아두이노 연결)
try:
    arduino = serial.Serial('COM9', 9600)  # COM 포트는 실제 연결된 포트로 변경
except Exception as e:
    print(f"Error connecting to Arduino: {e}")
    arduino = None

# 특정 클래스 설정
target_classes = ["person", "persons", "car", "cars"]  # 판단 대상
vehicle_signal = "GREEN"  # 차량 신호 상태 (초기값)
pedestrian_signal = "GREEN"  # 보행자 신호 상태 (초기값)
vehicle_yellow_start_time = None  # 차량 노란불 시작 시간
pedestrian_yellow_start_time = None  # 보행자 노란불 시작 시간
last_person_detected_time = None  # 마지막으로 사람이 감지된 시간


def send_command_to_arduino(command):
    """아두이노로 명령 전송"""
    if arduino:
        arduino.write(f"{command}\n".encode())
        print(f"Sent to Arduino: {command}")
    else:
        print(f"Arduino not connected. Command skipped: {command}")


def switch_vehicle_signal(new_signal):
    """차량 신호 전환 (노란불 포함)"""
    global vehicle_signal, vehicle_yellow_start_time

    if vehicle_signal != new_signal:
        # 노란불 타이머 설정
        if vehicle_signal == "GREEN" and new_signal == "RED" and vehicle_yellow_start_time is None:
            send_command_to_arduino("GREEN3_OFF")
            send_command_to_arduino("YELLOW3_ON")
            vehicle_yellow_start_time = time.time()
            return
        if vehicle_signal == "RED" and new_signal == "GREEN" and vehicle_yellow_start_time is None:
            send_command_to_arduino("RED3_OFF")
            send_command_to_arduino("YELLOW3_ON")
            vehicle_yellow_start_time = time.time()
            return

        # 노란불 유지 후 신호 전환
        if vehicle_yellow_start_time and time.time() - vehicle_yellow_start_time >= 1:
            send_command_to_arduino("YELLOW3_OFF")
            if new_signal == "RED":
                send_command_to_arduino("RED3_ON")
            elif new_signal == "GREEN":
                send_command_to_arduino("GREEN3_ON")
            vehicle_signal = new_signal
            vehicle_yellow_start_time = None  # 타이머 초기화


def switch_pedestrian_signal(new_signal):
    """보행자 신호 전환 (노란불 포함)"""
    global pedestrian_signal, pedestrian_yellow_start_time

    if pedestrian_signal != new_signal:
        # 노란불 타이머 설정
        if pedestrian_signal == "GREEN" and new_signal == "RED" and pedestrian_yellow_start_time is None:
            send_command_to_arduino("GREEN1_OFF,GREEN2_OFF")
            send_command_to_arduino("YELLOW1_ON,YELLOW2_ON")
            pedestrian_yellow_start_time = time.time()
            return

        # 노란불 유지 후 신호 전환
        if pedestrian_yellow_start_time and time.time() - pedestrian_yellow_start_time >= 1:
            send_command_to_arduino("YELLOW1_OFF,YELLOW2_OFF")
            if new_signal == "RED":
                send_command_to_arduino("RED1_ON,RED2_ON")
            elif new_signal == "GREEN":
                send_command_to_arduino("GREEN1_ON,GREEN2_ON")
            pedestrian_signal = new_signal
            pedestrian_yellow_start_time = None  # 타이머 초기화


def process_video_with_roi_logic(input_path, output_path, roi_coords):
    global vehicle_signal, pedestrian_signal, last_person_detected_time

    # 비디오 읽기
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {input_path}")
        return

    x1, y1, x2, y2 = roi_coords
    roi_width = x2 - x1
    roi_height = y2 - y1

    print("Processing video...")

    # FPS 및 출력 설정
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (roi_width, roi_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ROI 설정 (지정된 영역 추출)
        roi = frame[y1:y2, x1:x2]

        # YOLOv8로 객체 탐지
        results = model(roi, conf=0.5)
        detected_classes = set()

        # 탐지된 클래스 추출 및 Bounding Box 그리기
        for box in results[0].boxes.data:
            x1_box, y1_box, x2_box, y2_box, conf, cls = box
            x1_box, y1_box, x2_box, y2_box = map(int, [x1_box, y1_box, x2_box, y2_box])
            class_name = model.names[int(cls)]
            detected_classes.add(class_name)

            # Bounding Box 그리기
            label = f"{class_name} {conf:.2f}"
            cv2.rectangle(roi, (x1_box, y1_box), (x2_box, y2_box), (0, 255, 0), 2)
            cv2.putText(roi, label, (x1_box, y1_box - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 차량 및 보행자 감지 여부
        car_detected = any(cls in ["car", "cars"] for cls in detected_classes)
        person_detected = any(cls in ["person", "persons"] for cls in detected_classes)

        # 로직 처리
        current_time = time.time()
        if person_detected and not car_detected:
            # 보행자 감지, 차량 미감지
            last_person_detected_time = current_time
            switch_pedestrian_signal("GREEN")
            switch_vehicle_signal("RED")
        elif car_detected and person_detected:
            # 차량, 보행자 동시 감지
            switch_pedestrian_signal("GREEN")
            switch_vehicle_signal("RED")
        elif not person_detected and car_detected:
            # 차량 감지, 보행자 미감지
            switch_pedestrian_signal("RED")
            switch_vehicle_signal("GREEN")
        else:
            # 아무도 감지되지 않을 때
            switch_pedestrian_signal("RED")
            switch_vehicle_signal("GREEN")

        # ROI 부분 저장 및 출력
        out.write(roi)
        cv2.imshow("ROI Detection", roi)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키로 종료
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processed video saved to: {output_path}")


if __name__ == "__main__":

    # Example 1
    input_video = "input/video_1.mp4"
    output_video = "output/video_1.mp4"

    # ROI 설정    
    roi_coordinates = (1090, 600, 1750, 950)

    # # Example 2
    # input_video = "input/video_2.mp4"
    # output_video = "output/video_2.mp4"

    # # ROI 설정
    # roi_coordinates = (75, 150, 150, 900)



    # # Example 3
    # input_video = "input/video_3.mp4"
    # output_video = "output/video_3.mp4"

    # # ROI 설정
    # roi_coordinates = (75, 150, 500, 650)

    # 디렉토리 생성
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    process_video_with_roi_logic(input_video, output_video, roi_coordinates)
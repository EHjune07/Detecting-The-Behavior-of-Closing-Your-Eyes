import cv2
import numpy as np
from mtcnn import MTCNN
import time
import pygame

# Pygame 초기화
pygame.mixer.init()

# MTCNN 초기화
detector = MTCNN()

# Haar Cascade 분류기 로드 (눈 인식을 위한)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 눈 깜빡임 카운트 변수 및 타이머 초기화
count = 0                  # count 졸음 3초 - count +1 증가
warning_count = 0          # warning_count 졸음 5초 - warning_count +1 증가 
eye_closed_time = 0        # count 전용 졸음 시간
warning_time = 0           # warning_count 전용 졸음 시간
is_eye_closed = False      # 현재 눈이 감겨 있는지 여부를 나타내는 변수

# 비디오 캡처 시작 (카메라 지정 및 해상도 설정)
capture = cv2.VideoCapture(0)                   # 카메라 번호 설정(n)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 750)      # 가로 해상도 설정
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)     # 세로 해상도 설정

while True:
    ret, frame = capture.read()     # 비디오 프레임 읽기(카메라 켜기)
    if not ret:                     # 프레임이 유효하지 않으면 루프 종료
        break

    results = detector.detect_faces(frame)  # 얼굴 탐지

    if results:                                                                 # 탐지된 얼굴이 있으면
        x, y, width, height = results[0]['box']                                 # 첫 번째 얼굴의 위치와 크기 가져오기
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)   # 인식된 얼굴에 프레임 출력 (BGR - G)

        upper_face = frame[y:y + height // 2, x:x + width]                                      # 프레임의 상반부만 선택 (콧구멍, 입을 눈으로 인식하는 것 방지)
        gray_upper_face = cv2.cvtColor(upper_face, cv2.COLOR_BGR2GRAY)                          # 프레임의 상반부를 그레이스케일(흑백)로 변환(최적화)
        eyes = eye_cascade.detectMultiScale(gray_upper_face, scaleFactor=1.1, minNeighbors=5)   # 눈 탐지

        if len(eyes) == 0:                          # 감지된 눈이 없다면(눈이 감겨 있으면)
            if not is_eye_closed:                   # 눈이 처음 감겨졌다면
                eye_closed_time = time.time()       # 현재 시간을 eye_closed_time에 저장
                warning_time = eye_closed_time      # 경고 시간 초기화
                is_eye_closed = True                # 눈이 감겼음을 표시

            elapsed_time = time.time() - eye_closed_time            # 눈을 감은 시간 계산
            warning_elapsed_time = time.time() - warning_time       # 경고 시간을 계산

            # count 증가 조건
            if elapsed_time >= 3:               # 3초 이상 눈이 감겨있으면
                count += 1                      # count에 +1
                eye_closed_time = time.time()   # 눈 깜빡임 방지 (연속된 count 증가 방지)

                # count 조건에 따른 mp3 재생
                if count == 3:                              # 짧은(3초간) 3회 눈감음
                    pygame.mixer.music.load('1단계.mp3')    # 1단계 mp3 불러오기
                    pygame.mixer.music.play()               # 불러온 MP3 재생 - 1

                elif count == 4:                            # 짧은 4회 눈감음
                    pygame.mixer.music.load('2단계.mp3')    # 2단계 mp3 불러오기
                    pygame.mixer.music.play()               # 불러온 mp3 재생 - 2

                elif count == 5:                            # 짧은 5회 눈감음
                    pygame.mixer.music.load('3단계.wav')    # 3단계 wav 불러오기 (.wav 확장자 명 변경했는데 오류로 실행 안됨)
                    pygame.mixer.music.play()               # 불러온 wav 재생 - 3

                elif count >= 6:                            # 짧은 6회 이상 눈감음(여기부턴 졸음쉼터에서 자야함.)
                    pygame.mixer.music.load('5단계.mp3')    # 5단계 mp3 불러오기
                    pygame.mixer.music.play()               # 불러온 mp3 재생 - 5

            # warning_count 증가 조건
            if warning_elapsed_time >= 5:           # 5초 이상 눈을 감고 있으면
                warning_count += 1                  # 경고 카운트 증가
                warning_time = time.time()          # 눈 깜빡임 방지 (연속된 warning_count 방지)

                # warning_count 조건에 따른 mp3 재생
                if warning_count == 1:                      # 장기간(5초간) 눈감음 1회
                    pygame.mixer.music.load('4단계.mp3')    # 4단계 mp3 불러오기
                    pygame.mixer.music.play()               # 불러온 mp3 재생 - 4
                
                elif warning_count == 2:                    # 장기간 눈감음 2회
                    pygame.mixer.music.load('5단계.mp3')    # 5단계 mp3 불러오기
                    pygame.mixer.music.play()               # 불러온 mp3 재생 - 5

                elif warning_count >= 3:                    # 장기간 눈감음 3회 이상(졸음쉼터 직행)
                    pygame.mixer.music.load('5단계.mp3')    # 5단계 mp3 불러오기
                    pygame.mixer.music.play()               # 불러온 mp3 재생 - 5

        else:                       # 눈이 감기지 않으면
            is_eye_closed = False   # 눈이 감기지 않음을 표시

            for (ex, ey, ew, eh) in eyes:                                                   # 탐지된 모든 눈에 대해
                cv2.rectangle(upper_face, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)     # 탐지된 눈 주변에 시각적 프레임 출력(BGR - R)

    # count 스코어와 warning_count 스코어를 시각적으로 텍스트 출력
    # (frame(텍스트를 표시할 프레임), f"카운트: {카운트 변수}", (x,y - 텍스트 위치), 폰트, 텍스트 크기, BGR, 두께)
    cv2.putText(frame, f"Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)                                # count 텍스트 출력 및 텍스트 설정
    cv2.putText(frame, f"Warning: {warning_count}", (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)    # warning_count 텍스트 출력 및 텍스트 설정

    cv2.imshow("detection of drowsiness", frame)  # 실행 창 이름

    if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키를 누르면 루프 종료
        break

capture.release()           # 비디오 캡처 해제
cv2.destroyAllWindows()     # 모든 OpenCV 창 닫기
import cv2

# 보통 카메라 ID는 0부터 5 사이입니다.
for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        print(f"성공: 카메라를 찾았습니다! ID: {i}")
        cap.release()
    else:
        print(f"실패: 카메라 ID {i} 에서는 카메라를 찾을 수 없습니다.")
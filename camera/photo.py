import cv2
import os

# camCheck.py를 통해 확인된 정확한 카메라 ID
CAM_LEFT_ID = 0
CAM_RIGHT_ID = 1

# 저장할 폴더 이름
SAVE_DIR_LEFT = "captures/left"
SAVE_DIR_RIGHT = "captures/right"

# 저장 폴더가 없으면 생성
os.makedirs(SAVE_DIR_LEFT, exist_ok=True)
os.makedirs(SAVE_DIR_RIGHT, exist_ok=True)


# [수정된 부분] 카메라 제어 방식을 DSHOW -> MSMF 로 변경
# DSHOW에서 동시 연결 문제가 발생할 때 MSMF가 해결책이 될 수 있습니다.
cap1 = cv2.VideoCapture(CAM_LEFT_ID, cv2.CAP_MSMF)
cap2 = cv2.VideoCapture(CAM_RIGHT_ID, cv2.CAP_MSMF)

# 웹캠이 정상적으로 열렸는지 확인
if not cap1.isOpened():
    print(f"오류: 1번 카메라 (ID: {CAM_LEFT_ID})를 열 수 없습니다. 다른 프로그램을 종료하거나 카메라를 재연결해보세요.")
    exit()
if not cap2.isOpened():
    print(f"오류: 2번 카메라 (ID: {CAM_RIGHT_ID})를 열 수 없습니다. 다른 프로그램을 종료하거나 카메라를 재연결해보세요.")
    exit()

print("카메라가 준비되었습니다.")
print("스페이스바를 눌러 사진을 촬영하세요.")
print("'q' 키를 누르면 프로그램이 종료됩니다.")

# 창 이름 설정
win_left = "Camera 1 (Left)"
win_right = "Camera 2 (Right)"
cv2.namedWindow(win_left)
cv2.namedWindow(win_right)


while True:
    # 각 웹캠에서 프레임 읽기
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    # 프레임을 성공적으로 읽어왔는지 확인
    if not ret1 or not ret2:
        print("오류: 카메라에서 프레임을 읽어오는 데 실패했습니다.")
        break

    # 화면에 웹캠 영상 표시
    cv2.imshow(win_left, frame1)
    cv2.imshow(win_right, frame2)

    # 키보드 입력 대기 (1ms)
    key = cv2.waitKey(1) & 0xFF

    # 'q' 키를 누르면 루프 종료
    if key == ord('q'):
        print("프로그램을 종료합니다.")
        break

    # 스페이스바를 누르면 사진 저장
    if key == ord(' '):
        # 각 폴더의 파일 개수를 세어 새 파일 번호 생성
        left_num = len(os.listdir(SAVE_DIR_LEFT)) + 1
        right_num = len(os.listdir(SAVE_DIR_RIGHT)) + 1

        # 파일 경로 설정
        left_filename = os.path.join(SAVE_DIR_LEFT, f"lm_L_{left_num}.png")
        right_filename = os.path.join(SAVE_DIR_RIGHT, f"lm_R_{right_num}.png")

        # 이미지 저장
        cv2.imwrite(left_filename, frame1)
        cv2.imwrite(right_filename, frame2)

        print(f"성공! 이미지가 저장되었습니다: \n  - {left_filename}\n  - {right_filename}")


# 모든 작업이 끝나면 자원 해제
cap1.release()
cap2.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import os
from tqdm import tqdm

# --- 함수 정의 (수정 없음) ---

def find_chessboard_corners(image: np.ndarray, chessboard_size: tuple) -> np.ndarray:
    """주어진 이미지에서 체스보드 코너의 2D 픽셀 좌표를 찾습니다."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    found, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if not found:
        return None

    corners_subpix = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1), criteria
    )
    return corners_subpix

def visualize_and_number_corners(image: np.ndarray, corners: np.ndarray, chessboard_size: tuple) -> np.ndarray:
    """이미지에 검출된 코너와 번호를 그려 시각화합니다."""
    cv2.drawChessboardCorners(image, chessboard_size, corners, True)
    for i, corner in enumerate(corners):
        x, y = int(corner[0][0]), int(corner[0][1])
        cv2.putText(
            image, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
            0.4, (0, 0, 255), 1, cv2.LINE_AA
        )
    return image

# --- 메인 실행 함수 ---

def main():
    """메인 실행 함수"""
    # ===================================================================
    # 사용자 설정 영역 (이 부분의 값을 직접 수정하세요)
    # ===================================================================
    input_path = "my_images"
    output_txt_path = "corner_coordinates"
    output_viz_path = "corner_visualizations"
    chessboard_size = (9, 6)
    # ===================================================================

    os.makedirs(output_txt_path, exist_ok=True)
    os.makedirs(output_viz_path, exist_ok=True)

    try:
        image_files = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    except FileNotFoundError:
        print(f"🚨 오류: 입력 폴더 '{input_path}'를 찾을 수 없습니다!")
        return

    if not image_files:
        print(f"⚠️ 경고: 입력 폴더 '{input_path}'에 이미지 파일이 없습니다.")
        return

    print(f"총 {len(image_files)}개의 이미지에서 코너 검출을 시작합니다...")

    # --- 카운터 및 실패 목록 초기화 ---
    success_count = 0
    failure_count = 0
    failed_files = []

    # 메인 루프
    for image_name in tqdm(image_files, desc="진행 상황"):
        image_path = os.path.join(input_path, image_name)
        image = cv2.imread(image_path)

        if image is None:
            failure_count += 1
            failed_files.append(f"{image_name} (파일 읽기 실패)")
            continue

        corners = find_chessboard_corners(image.copy(), chessboard_size)

        if corners is not None:
            # --- 검출 성공 시 ---
            success_count += 1
            base_name = os.path.splitext(image_name)[0]

            # 1. 좌표 텍스트 파일 저장 (인덱스 추가)
            output_filepath = os.path.join(output_txt_path, f"{base_name}.txt")
            
            # (Nx2) 형태로 변환
            corners_reshaped = corners.reshape(-1, 2)
            # 0부터 시작하는 인덱스 열 생성
            indices = np.arange(len(corners_reshaped)).reshape(-1, 1)
            # 인덱스와 좌표 데이터를 수평으로 합침
            data_to_save = np.hstack((indices, corners_reshaped))
            
            # 텍스트 파일로 저장 (포맷 및 헤더 수정)
            np.savetxt(
                output_filepath,
                data_to_save,
                fmt=['%d', '%.6f', '%.6f'],  # 포맷: [정수, 실수, 실수]
                delimiter=',',
                header='corner_index,x,y',  # 헤더: [코너번호, x좌표, y좌표]
                comments=''
            )

            # 2. 시각화 이미지 생성 및 저장
            viz_image = visualize_and_number_corners(image, corners, chessboard_size)
            output_image_path = os.path.join(output_viz_path, f"{base_name}_corners.jpg")
            cv2.imwrite(output_image_path, viz_image)
        else:
            # --- 검출 실패 시 ---
            failure_count += 1
            failed_files.append(image_name)

    # --- 최종 결과 출력 ---
    print("\n" + "="*40)
    print("✅ 모든 작업이 완료되었습니다.")
    print(f"처리 결과: 총 {len(image_files)}개 이미지 중 {success_count}개 성공, {failure_count}개 실패")
    print(f"📂 좌표 파일 저장 위치: '{output_txt_path}'")
    print(f"🖼️ 시각화 이미지 저장 위치: '{output_viz_path}'")
    
    if failed_files:
        print("\n 실패한 파일 목록:")
        for fname in failed_files:
            print(f" - {fname}")
    print("="*40)


if __name__ == "__main__":
    main()
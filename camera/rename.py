import os
import glob

# 캡처된 이미지들이 저장된 폴더 경로
# 스크립트 파일과 같은 위치에 captures 폴더가 있다면 그대로 두세요.
captures_dir = 'captures'

def rename_capture_files(target_dir):
    """
    지정된 폴더 안의 left_*, right_* 이미지 파일 이름을
    lm_L_번호, lm_R_번호 형식으로 변경합니다.
    """
    if not os.path.exists(target_dir):
        print(f"오류: '{target_dir}' 폴더를 찾을 수 없습니다.")
        return

    # 1. left와 right 이미지 파일 목록을 각각 가져와서 시간순으로 정렬합니다.
    # 파일 이름에 타임스탬프가 있으므로 이름순으로 정렬하면 시간순이 됩니다.
    left_files = sorted(glob.glob(os.path.join(target_dir, 'left_*.png')))
    right_files = sorted(glob.glob(os.path.join(target_dir, 'right_*.png')))

    if not left_files or not right_files:
        print("이름을 변경할 파일을 찾지 못했습니다. 파일 이름이 'left_' 또는 'right_'로 시작하는지 확인하세요.")
        return
        
    if len(left_files) != len(right_files):
        print("경고: left 이미지와 right 이미지의 개수가 일치하지 않습니다.")
        # 더 적은 쪽의 개수에 맞춰서 진행
        min_count = min(len(left_files), len(right_files))
        left_files = left_files[:min_count]
        right_files = right_files[:min_count]
        print(f"{min_count} 쌍의 파일만 처리합니다.")

    # 2. 변경될 파일 이름 계획 세우기 (Dry Run)
    rename_plan = []
    for i in range(len(left_files)):
        # 새 파일 이름 생성 (번호는 1부터 시작)
        new_left_name = f"lm_L_{i+1}.png"
        new_right_name = f"lm_R_{i+1}.png"

        # 원래 경로와 새 경로 준비
        old_left_path = left_files[i]
        old_right_path = right_files[i]
        new_left_path = os.path.join(target_dir, new_left_name)
        new_right_path = os.path.join(target_dir, new_right_name)
        
        rename_plan.append((old_left_path, new_left_path))
        rename_plan.append((old_right_path, new_right_path))

    # 3. 사용자에게 변경 계획을 보여주고 확인 받기
    print("--- 파일 이름 변경 계획 (실제 변경 전 미리보기) ---")
    for old, new in rename_plan:
        print(f"'{os.path.basename(old)}'  ->  '{os.path.basename(new)}'")
    
    print("-" * 50)
    
    try:
        confirm = input("위와 같이 파일 이름 변경을 진행하시겠습니까? (y/n): ")
    except KeyboardInterrupt:
        print("\n작업이 취소되었습니다.")
        return

    # 4. 'y'를 입력했을 경우에만 실제 이름 변경 실행
    if confirm.lower() == 'y':
        try:
            for old, new in rename_plan:
                os.rename(old, new)
            print("\n성공적으로 파일 이름을 변경했습니다!")
        except OSError as e:
            print(f"\n파일 이름 변경 중 오류가 발생했습니다: {e}")
    else:
        print("\n작업이 취소되었습니다.")


if __name__ == '__main__':
    rename_capture_files(captures_dir)
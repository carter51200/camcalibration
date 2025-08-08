## camCalibration

체스보드 패턴을 이용해 카메라(단안/스테레오) 보정과 영상 정렬을 수행하는 프로젝트입니다. 예제 데이터와 보정 결과는 `data/` 폴더 구조를 참고하세요.

### 사전 준비
- Python 설치 후 의존성 설치:

```bash
pip install -r requirements.txt
```

### 실행 방법
아래 명령어로 보정을 실행합니다. 경로 기준은 프로젝트 루트입니다.

```bash
python main.py --input_path data --chessboard_size 9,6 --square_size 0.021
```

### 주요 인자 설명
- **--input_path**: 입력 데이터가 들어있는 디렉터리 경로 (예: `data`)
- **--chessboard_size**: 체스보드의 내부 코너 수를 `가로,세로` 순서로 지정 (예: `9,6`)
- **--square_size**: 체스보드 한 칸의 한 변 길이(미터 단위). 예: `0.021` → 21mm

### 참고
- 예제 폴더 구조:
  - 원본 이미지: `data/images/left`, `data/images/right`
  - 코너 시각화: `data/corners/left`, `data/corners/right`
  - 보정 파라미터: `data/params/`
  - 보정/정렬 결과: `data/rectified/left`, `data/rectified/right`

문제가 발생하면 Python 버전과 의존성 설치 여부를 확인한 뒤, 실행 로그의 에러 메시지를 첨부하여 이슈를 남겨주세요.



# Deep Learning App Tutorial
이 레포지토리는 딥러닝 모델 및 이를 활용한 어플리케이션 개발 과정을 단계별로 설명하는 튜토리얼을 제공합니다.

## 설치 환경
python version: 3.12

## 설치 방법
1. uv 설치: https://github.com/astral-sh/uv 참고
    ```
    python -m pip install uv
    ```
1. python 3.12.3 설치
    ```
    python -m uv python install 3.12.3
    ```
1. 가상 환경 생성
    ```
    python -m uv venv --python 3.12.3
    ```
    
1. 가상 환경 실행
    - mac / linux
    ```
    source .venv/bin/activate
    ```
    - windows
    ```
     .\.venv\Scripts\activate
    ```
1. 필수 패키지 설치
    ```
    python -m pip install uv
    uv pip install -r requirements.txt
    ```
1. 환경 변수 파일 생성(.env.example 파일을 복사하여 .env 파일로 저장 후 적절한 값을 입력)
    ```
    WORKING_DIRECTORY=/Users/jeanboy/workspace/ml-app-tutorial
    PYTHONPATH=/Users/jeanboy/workspace/ml-app-tutorial
    MLFLOW_URI=file:///Users/jeanboy/workspace/ml-app-tutorial/runs/mlflow
    ```

## 실습 순서
0. 환경 설정 및 확인
1. streamlit 배우기
1. 객체탐지모델을 활용한 음식점 리뷰 서비스 고도화
    1. yolov5 맛보기
    1. Streamlit app으로 서비스 프로토타입 만들기
    1. ML 모델 만들기(모델 학습 및 평가)
    1. 모델 성능 개선 - Mlflow 적용하기
    1. Streamlit app 개선하기
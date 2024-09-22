# Deep Learning App Tutorial
이 레포지토리는 딥러닝 모델 및 이를 활용한 어플리케이션 개발 과정을 단계별로 설명하는 튜토리얼을 제공합니다.

## 설치 환경
python version: 3.12

## 설치 방법
1. uv 설치: https://github.com/astral-sh/uv 참고
1. python 3.12.3 설치
    ```
    uv python install 3.12.3
    ```
1. 가상 환경 생성
    ```
    uv venv --python 3.12.3
    ```
1. 가상 환경 실행
    ```
    source .venv/bin/activate
    ```
1. 필수 패키지 설치
    ```
    uv pip install -r requirements.txt
    ```
1. utralytics/yolov5 소스코드 및 관련 패키지 설치
    ```
    git clone https://github.com/ultralytics/yolov5
    cd yolov5
    uv pip install -r requirements.txt  # install
    ```

## 실습 순서
1. streamlit 배우기
1. 객체탐지모델을 활용한 음식점 리뷰 서비스 고도화
    1. yolov5 맛보기
    1. Streamlit app으로 서비스 프로토타입 만들기
    1. ML 모델 만들기(모델 학습 및 평가)
    1. 모델 성능 개선 - Mlflow 적용하기
    1. 모델 성능 비교 App 만들기
    1. 추가 음식에 대한 학습,배포 자동화하기
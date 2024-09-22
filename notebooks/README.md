# 노트북
다양한 분석 모델을 실험하기 위한 코드를 저장합니다.

## 코드 실행
### sample.py
샘플 노트북을 실행합니다. 아래의 코드를 실행하기 위해서는 `README.md`의 가이드를 따라 환경설정을 해야합니다.
```
python notebooks/sample.py
```

### yolov5.py
yolov5 를 사용해서 예측하고 결과를 저장합니다. 코드를 실행하면 실행한 경로 내에 `yolov5s.pt` 모델을 다운로드 하고 예측 결과를 `runs/detect/exp` 아래 저장합니다.
```
python notebooks/yolov5.py
```

### train_base_model.py
기본 모델을 학습합니다. 모델 학습 전 아래의 과정을 통해 필요한 데이터셋을 준비해야 합니다.

1. 학습, 평가, 테스트 데이터 생성
    [../src/README.md](../src/README.md) 파일의 `filter_data.py` 스크립트 사용법을 참고하여 학습, 평가, 테스트 데이터를 생성합니다.

1. `data.yaml` 수정
    아래의 예시를 참고하여 train, val, test 데이터의 경로를 입력합니다.
    ```
    train: ../train/images
    val: ../valid/images
    test: ../test/images

    nc: 2
    names: [Jjambbong, jjajangmyeon]

    ```

1. 모델 학습
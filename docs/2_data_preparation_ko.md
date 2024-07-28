# Data Preparation
여기서는 Transformer 챗봇 모델 훈련 튜토리얼을 위해 다음 카페의 ["사랑보다 아름다운 실연"]( http://cafe116.daum.net/_c21_/home?grpid=1bld) Counselor 데이터셋을 사용합니다.
Custom 데이터를 이용하기 위해서는 아래 설명을 참고하시기 바랍니다.

### 1. Counselor Data
Counselor 데이터를 학습하고싶다면 아래처럼 `config/config.yaml` 파일의 `counselor_dataset_train` 값을 `True`로 설정하면 됩니다.
```yaml
counselor_dataset_train: True   # If True, counselor data will be loaded automatically.
counselor_dataset:
    path: data/
CUSTOM:
    train_data_path: null
    validation_data_path: null
    test_data_path: null
```
<br>

### 2. Custom Data
만약 custom 데이터를 학습하고 싶다면 아래처럼 `config/config.yaml`의 `counselor_dataset_train` 값을 `False` 설정하면 됩니다.
다만 `src/trainer/build.py`에 custom dataset 사용을 위한 코드를 추가로 작성해야합니다.
```yaml
counselor_dataset_train: False   # If True, counselor data will be loaded automatically.
counselor_dataset:
    path: data/
CUSTOM:
    train_data_path: null
    validation_data_path: null
    test_data_path: null
```
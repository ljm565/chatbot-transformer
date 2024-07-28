# Data Preparation
Here, we will proceed with a Transformer chatbot model training tutorial using the Daum Cafe ["사랑보다 아름다운 실연"]( http://cafe116.daum.net/_c21_/home?grpid=1bld) dataset by default.
Please refer to the following instructions to utilize custom datasets.

### 1. Counselor Data
If you want to train on the counselor dataset, simply set the `counselor_dataset_train` value in the `config/config.yaml` file to `True` as follows.
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
If you want to train your custom dataset, set the `counselor_dataset_train` value in the `config/config.yaml` file to `False` as follows.
You may require to implement your custom dataloader codes in `src/utils/data_utils.py`.
```yaml
counselor_dataset_train: False   # If True, counselor data will be loaded automatically.
counselor_dataset:
    path: data/
CUSTOM:
    train_data_path: null
    validation_data_path: null
    test_data_path: null
```
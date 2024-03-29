# Korean Chatbot Transformer
## 설명
다음 카페 ["사랑보다 아름다운 실연"]( http://cafe116.daum.net/_c21_/home?grpid=1bld) 데이터를 사용하여 [송영숙](https://github.com/songys/Chatbot_data)님께서 만든 챗봇 데이터를 통해 챗봇 모델을 학습합니다.
Transformer 기반 기계 번역 모델에 대한 설명은 [Transformer를 이용한 한국어 대화 챗봇](https://ljm565.github.io/contents/transformer3.html)을 참고하시기 바랍니다.
또한 본 모델은 vanilla transformer에서 사용하는 positional encoding 뿐만 아니라, positional embedding을 선택할 수 있습니다.
마지막으로 최종 학습된 모델을 바탕으로 실제 챗봇을 구동해볼 수 있습니다.
<br><br><br>

## 모델 종류
* ### Transformer
    한국어 대화 챗봇 모델 제작을 위해 transformer를 학습합니다.
<br><br><br>


## 토크나이저 종류
* ### Wordpiece Tokenizer
    Likelihood 기반으로 BPE를 수행한 한국어 subword 토크나이저를 사용합니다.
    학습을 한 번 실행 시킨 후, 토크나이저가 없다는 메시지를 뱉을 시 wordpiece 토크나이저의 vocab 파일을 모델 학습 전에 아래 명령어를 이용하여 먼저 제작해야합니다. 제작할 vocab의 크기는 src/tokenizer/make_vocab.sh에서 수정할 수 있습니다(Default: 8,000).
    
    ```
    cd src/tokenizer
    bash ./make_vocab.sh
    ```
<br><br>

## 사용 데이터
다음 카페 ["사랑보다 아름다운 실연"]( http://cafe116.daum.net/_c21_/home?grpid=1bld) 데이터를 사용하여 [송영숙](https://github.com/songys/Chatbot_data)님께서 만든 챗봇 데이터를 사용합니다.
<br><br><br>


## 사용 방법
* ### 학습 방법
    **코드 테스트를 해보기 전, 먼저 데이터 폴더 이름을 위의 '사용 데이터' 부분에 설명한 것 처럼 바꿔줘야 합니다.**
    바꿔준 후, 학습을 시작하기 위한 argument는 4가지가 있습니다.<br>
    * [-d --device] {cpu, gpu}, **필수**: 학습을 cpu, gpu로 할건지 정하는 인자입니다.
    * [-m --mode] {train, inference, chatting}, **필수**: 학습을 시작하려면 train, 학습된 모델을 가지고 있어서 대화 샘플 결과를 보고싶은 경우에는 inference로 설정해야합니다. 챗봇을 구동해보기 위해서는 chatting 인자를 사용하면 됩니다.
    inference 모드를 사용할 경우, [-n, --name] 인자가 **필수**입니다.
    * [-c --cont] {1}, **선택**: 학습이 중간에 종료가 된 경우 다시 저장된 모델의 체크포인트 부분부터 학습을 시작할 수 있습니다. 이 인자를 사용할 경우 -m train 이어야 합니다. 
    * [-n --name] {name}, **선택**: 이 인자는 -c 1 혹은 -m {inference} 경우 사용합니다.
    중간에 다시 불러서 학습을 할 경우 모델의 이름을 입력하고, inference를 할 경우에도 실험할 모델의 이름을 입력해주어야 합니다(최초 학습시 src/config.json에서 정한 모델의 이름의 폴더가 형성되고 그 폴더 내부에 모델 및 모델 파라미터가 json 파일로 형성 됩니다).<br><br>

    터미널 명령어 예시<br>
    * 최초 학습 시
        ```
        python3 src/main.py -d cpu -m train
        ```
    * 중간에 중단 된 모델 이어서 학습 시
        <br>주의사항: config.json을 수정해야하는 일이 발생 한다면 base_path/src/config.json이 아닌, base_path/src/model/{model_name}/{model_name}.json 파일을 수정해야 합니다.
        ```
        python3 src/main.py -d gpu -m train -c 1 -n {model_name}
        ```
    * 최종 학습 된 모델의 test set에 대한 몇가지 샘플 결과를 확인할 시
        <br>주의사항: config.json을 수정해야하는 일이 발생 한다면 base_path/src/config.json이 아닌, base_path/src/model/{model_name}/{model_name}.json 파일을 수정해야 수정사항이 반영됩니다.
        ```
        python3 src/main.py -d cpu -m inference -n {model_name}
        ```
    <br><br>

* ### 모델 학습 조건 설정 (config.json)
    **주의사항: 최초 학습 시 config.json이 사용되며, 이미 한 번 학습을 한 모델에 대하여 parameter를 바꾸고싶다면 base_path/src/model/{model_name}/{model_name}.json 파일을 수정해야 합니다.**
    * base_path: 학습 관련 파일이 저장될 위치.
    * model_name: 학습 모델이 저장될 파일 이름 설정. 모델은 base_path/src/model/{model_name}/{model_name}.pt 로 저장.
    * loss_data_name: 학습 시 발생한 loss data를 저장하기 위한 이름 설정. base_path/src/loss/{loss_data_name}.pkl 파일로 저장. 내부에 중단된 학습을 다시 시작할 때, 학습 과정에 발생한 loss 데이터를 그릴 때 등 필요한 데이터를 dictionary 형태로 저장.
    * vocab_size: 학습 시 사전 제작된 vocab 파일들 중 원하는 vocab size 선택. 만약 vocab_size를 8,000으로 설정 했을 시, data/tokenizer 폴더 내에 vocab_8000이라는 이름의 학습된 폴더가 있어야함.
    * max_len: 토큰화 된 번역 source, target 데이터의 최대 길이.
    * hidden_dim: Transformer 모델의 hidden dimension.
    * ffn_dim: Transformer 모델의 feed forward network의 hidden dimension.
    * enc_num_layers: Transformer encoder의 레이어 수.
    * dec_num_layers: Transformer decoder의 레이어 수.
    * num_head: Transformer attention head 수.
    * bias: {0, 1} 중 선택. 1이면 모델이 bias를 사용.
    * dropout: 모델의 dropout 비율.
    * layernorm_eps: Layer normalization epsilon 값.
    * pos_encoding: {0, 1} 중 선택. 1이면 positional encoding, 0이면 positional embedding 사용.
    * batch_size: batch size 지정.
    * epochs: 학습 epoch 설정.
    * lr: learning rate 지정.
    * early_stop_criterion: Validation set의 최대 BLEU-4를 내어준 학습 epoch 대비, 설정된 숫자만큼 epoch이 지나도 나아지지 않을 경우 학습 조기 종료.
    * result_num: 모델 테스트 시, 결과를 보여주는 sample 개수.
    <br><br><br>


## 결과
* ### 챗봇 학습 Score History
    * Validation Set BLEU History<br>
    <img src="images/img1.png" width="80%"><br><br>

    * Validation Set NIST History<br>
    <img src="images/img2.png" width="80%"><br><br>


* ### 챗봇 학습 결과 샘플
    chatting 모드로 실행하여 문장을 넣었을 때 나온 답변입니다.

    ```
    # Sample 1
    Q : 어디로 여행 가면 좋을까?
    A: 온 가족이 모두 마음에 드는 곳으로 가보세요.
    

    # Sample 2
    Q: 나 좋아하는 남자가 생겼어
    A: 충분히 그럴 수 있어여.


    # Sample 3
    Q: 오늘 저녁 뭐 먹을까?
    A: 맛있는 거 드세요.
    ```
<br><br><br>

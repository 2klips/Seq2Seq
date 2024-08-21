# **Sequence to Sequence**

## **Sequence to Sequence Learning with Neural Networks**

### **NIPS에서 발표 2014** [논문](https://arxiv.org/pdf/1409.3215)

---------------------

# **논문 요약**

## 초록

DNN은 어려운 학습에서도 뛰어난 성능을 보여주는 모델이나, 단점이 존재한다.
- 벡터의 크기가 고정되어 있어, 시퀀스데이터 끼리의 매핑이 불가능하다. ( 기계 번역, 음성 인식 )

해당 문제점을 해결하기 위해 LSTM을 사용하였으며,
기존의 SMT 모델과 정확도를 비교했을 때, LSTM를 사용했을 때 BLEU 점수가 높았으며, 단어의 순서를 바꾸었을 때, 더 높은 정확도가 나온다는 점을 발견

---------------------

# **1. Introduction**

### DNN에 대한 소개내용
- DNN은 음성인식, 시각적 객체 인식같은 어려운 학습에서 강력한 성을 보인다.
  - 병렬 계산을 수행
  - 충분한 데이터셋이 있을 때(지도 학습), 최적의 파라미터를 찾을 수 있음

#### DNN의 단점
- 고정된 크기의 벡터로 인코딩을 해야하는 제한이 있어, 길이가 다른 Seq와 Seq끼리의 매핑이 필요한 번역의 경우, 적합하지 않음 ( input과 target의 차원이 같아야함 )
- seq 데이터의 경우, 입력과 출력의 길이가 가변적이기 때문

### **해당 단점들을 해결하기 위해 LSTM을 사용한 Seq to Seq 모델을 제시**

하나의 LSTM을 사용하여 입력 시퀀스를 받아 고정된 차원의 벡터 (context vector)를 출력하고, 그 다음  또 하나의 LSTM를 사용하여 이전에 생성된 벡터를 입력(hidden state)으로 받아 출력 시퀀스를 추출한다.

<br>

<img src='https://mblogthumb-phinf.pstatic.net/MjAyMzAzMTBfMjkg/MDAxNjc4NDIxNjYwOTA3.I0RUYyGCOcG8pKmMKLsA-gn0OmtJ_Gjp0UwOpS40Tfog.skCk_AEPj54Ihm_rQWdH96um90QR0OtpPkhrukgqKrgg.PNG.kisooofficial/image.png?type=w800'>

<br>

이 그림에서 [A, B, C]를 입력 시퀀스로 받고, LSTM을 거쳐 벡터를 생성하고 그 벡터가 두 번째 LSTM으로 들어가(EOS) 시퀀스를 출력한다.

두 번째 LSTM에서 W를 입력받고, 앞서 받아온 context vector를 hidden state로 사용한다
<br>

\* 그림에선 A, B, C 로 입력되었지만, 실제로는 문장의 단어들을 역순으로 입력한다.

- 해당 아이디어로 학습을 진행 시, 더 좋은 정확도를 얻을 수 있었다더라
- 매우 긴 문장에서도 어려움 없이 처리할 수 있었다.


---------------------------------

# **2. Model**


## RNN의 모델

[ RNN의 수식 ] <br>
<img src='https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbym2dV%2FbtrvLBvE7UM%2F4R0MSrcs85g8nVfhXGUxtk%2Fimg.png'>
<br>

- RNN은 입력과 출력의 정렬을 미리 알고있으면, Seq와 Seq를 쉽게 매핑 할 수는 있음
- 하지만 입력과 출력의 길이가 다른 경우, 학습하기 어려움
- 장기 의존성 문제가 존재 ( 문장이 길 수록, 앞의 정보를 뒤까지 전달하기 어려움 )

해당 문제를 LSTM으로 해결하려 함


## LSTM

<img src='https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FznMbp%2FbtrGhVEhg8c%2FHkhaArlDiINV37cVurzPcK%2Fimg.png'>
<br>

x = 입력 시퀀스
y = 출력 시퀀스

LSTM은 조건부 확률을  p(y_1, y_2, ..., y_T' | x_1, x_2, ..., x_T) 계산
첫 번째 LSTM에서 출력한 context vector = v 를 hidden state로 y를 계산

- 논문에서 사용한 실제 모델은 두 개의 다른 LSTM을 사용함 ( 입력시퀀스, 출력시퀀스 )
- 4개의 레이어를 가진 LSTM을 사용
- 문장의 단어 순서를 역순으로 학습하는 것이 더 좋은 정확도가 나오는 것을 발견하여, 입력 문장의 단어들을 역순으로 입력함

------------------------------

# **3. Experiments ( 실험 )**

실험 데이터셋 : WMT'14 영어 > 프랑스어


### **디코딩 및 Rescoring**

훈련 목적 ->

<img src='https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FMBjj1%2FbtrwA5Kp89z%2FQLKP11Tk30KvVJ8XkFkDuk%2Fimg.png'>
<br>
문장 S가 입력됬을 때, 번역된 문장 T의 로그 확률을 최대화 해야함


### **문장 역순**
* 입력된 문장의 단어를 역순으로 처리하면 LSTM이 훨씬 더 잘 학습한다는 것을 발견함
* 장기 의존성 문제를 해결할 수 있다.

<br>

그 외 훈련에 사용한 학습 파라미터, 및 하드웨어 세팅에 대한 내용


### **실험결과**
- 기존 SMT보다 높은 점수를 기록함
<img src='https://cpm0722.github.io/assets/images/2020-05-10-Sequence-to-Sequence-Learning-with-Neural-Networks/02.jpg'>
<br>

- 긴 문장에서도 성능 저하 없이 작동함
<img src='https://cpm0722.github.io/assets/images/2020-05-10-Sequence-to-Sequence-Learning-with-Neural-Networks/05.jpg'>

---------------------------------

# **결론**

- LSTM이 기존 SMT의 성능을 능가한다.
- 충분히 데이터만 있다면, 다른 더 많은 시퀀스 학습 부분에서도 잘 작동할 것을 시사한다.
<br>
- ( 또나옴 ) 번역 부분에서 문장의 단어를 역순으로 학습하는 것이 더 높은 정확도가 나온 부분을 보고 놀람
- 긴 문장에서도 번역 정확도가 높게 나옴, 초기는 메모리 제한 때문에 실패할 것으로 예상했으나, 역순 데이터셋으로 학습한 LSTM은 긴 문장에서도 문제없이 정확하게 번역함

해당 LSTM 도입 방식을 통해, 추후, 번역 문제 말고도 다른 Seq to Seq 문제에서도 잘 작동할 수 있음을 시사한다.


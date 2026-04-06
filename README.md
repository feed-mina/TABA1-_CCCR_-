## 📘 TABA1-CCCR (AI 이미지넷 분류 예측)

ImageNet 모델링 생성 프로젝트 : 단국대학교와 한국클라우드컴퓨팅연구조합에서 진행하는 TABA 1기 마지막 프로젝트 



> 🏆 TABA 공모전 최우수상 수상 프로젝트

| 항목 | 내용 |
|------|------|
| **프로젝트 기간** | 2022.12.06 ~ 12.23 |
| **수행 기관** | 단국대학교, 한국클라우드컴퓨팅연구조합 |
| **사용 기술** | Python, Flask, PyTorch, Jupyter Notebook, Azure/AWS |

- **과제 주제**: ImageNet 데이터셋 기반 분류 정확도 예측
- **수행 방식**: 다양한 모델(ResNet, EfficientNet 등)의 성능 비교 및 시각화 분석
- **데이터 설명**:
  - 총 50,000장의 이미지 (1,000개 클래스)
  - Amazon Mechanical Turk 통해 수작업으로 정답 라벨링
- **성과**: 발표 평가에서 분석력, 결과 도출 능력을 인정받아 "최우수상" 수상


> "AI 모델 분석과 시각화를 이해하는 손을 키울 수 있는 기회였습니다."
> 기계학습의 흐름을 따라가며 각 모델 성능의 차이를 수치로 증명해보는 훈련이 되었고,
> 논문 기반 모델 학습, 팀 단위 실험 설계, 성과 발표까지 경험한 첫 프로젝트였습니다.
- **발표 자료 PDF**: [IMAGENET_8조_최종.pdf 다운로드](https://github.com/feed-mina/TABA1-_CCCR_-/raw/master/TABA_Presentation.pdf)


---

## 📁 프로젝트 구조

```
taba/
├── 📄 README.md                          # 프로젝트 설명 (현재 파일)
├── 📊 IMAGENET_8조_최종.pdf/pptx         # 최종 발표 자료
├── 📊 TABA_Presentation.pdf              # TABA 발표 자료
├── 📋 이미지넷 라벨표.xlsx                # 이미지넷 1,000개 클래스 라벨 정보
├── 📄 이미지넷_분류_모델_정리.pdf         # 주요 CNN 모델 아키텍처 정리 문서
│
├── 📚 논문 PDF (6개)                     # 주요 CNN 모델 원본 논문
│   ├── 1.ResNet.pdf
│   ├── 2.EfficientNet_.pdf
│   ├── 3.GoogleNet.pdf
│   ├── 4.AlexNet.pdf
│   ├── 5.MobileNet_.pdf
│   └── 6.VGGNet.pdf
│
├── 💻 code/                              # 실험 코드 (Jupyter Notebooks)
│   ├── 전처리_및 CNN 코드 설계하기 (0.8808).ipynb  ← 최종 발표 코드 ⭐
│   ├── lastproject.ipynb                 ← TensorFlow Hub 전이학습 실험
│   ├── alexnet.ipynb                     ← AlexNet 구현
│   ├── vggnet.ipynb                      ← VGGNet 구현
│   ├── GoogLeNetStudy.ipynb              ← GoogLeNet 학습 실험
│   ├── Sigmoid_vs_ReLU_vs_Tanh.ipynb    ← 활성화 함수 비교
│   ├── [Baseline]_1. TensorFlow를 활용한 이미지 분류.ipynb
│   ├── [Baseline]2._전이학습(transfer learning)을 통한 이미지 분류.ipynb
│   ├── imagenet_이미지확인.ipynb         ← 데이터 탐색
│   └── 날짜별 실험 (1208 ~ 1216)        ← 일자별 실험 기록
│
├── 🖼️ Image_dataset/                     # 이미지 데이터셋 (zip 압축)
│   ├── 5class_train_jpg.zip              # 5개 클래스 학습 데이터
│   ├── 5class_val_jpg.zip                # 5개 클래스 검증 데이터
│   ├── 5class_test_jpg.zip               # 5개 클래스 테스트 데이터
│   ├── train_dir.zip / validation_dir.zip
│   └── 201-400 클래스 구간 데이터
│
└── 📈 result/                            # 학습 결과 그래프
    ├── imagenet1_train_accuracy.png      # V1 학습 정확도 곡선
    ├── imagenet1_train_loss.png          # V1 학습 손실 곡선
    ├── imagenet2_testresult.png          # V2 테스트 결과
    └── imagenet2_trainresult.png         # V2 학습 결과
```

---

## 🔑 핵심 요약

| 항목 | 내용 |
|------|------|
| **데이터셋** | ImageNet 50,000장 (1,000 클래스) |
| **비교 모델** | ResNet, EfficientNet, GoogLeNet, AlexNet, MobileNet, VGGNet |
| **최고 정확도** | **0.8808** (전처리 및 CNN 코드 설계하기) |
| **데이터 분할** | 5클래스 기준 train/val/test 분리 실험 포함 |
| **결과물** | 학습 곡선(loss/accuracy) 시각화 PNG 4장 |

---

## 💻 발표 관련 핵심 코드 설명

### ⭐ 1. `전처리_및 CNN 코드 설계하기 (0.8808).ipynb` — 최종 발표 모델

> **최고 정확도 0.8808을 달성한 메인 코드** (발표 당일 사용)

#### 📌 주요 구성

| 단계 | 내용 |
|------|------|
| **GPU 설정** | TensorFlow GPU 사용 확인 (`/device:GPU:0`) |
| **데이터 로드** | `./data/train/` 경로에서 50,000장 이미지 로드 |
| **레이블 인코딩** | `LabelEncoder`로 폴더명 → 숫자 레이블 변환 |
| **Data Augmentation** | `ImageDataGenerator`로 데이터 증강 |
| **CNN 모델 설계** | 커스텀 CNN 아키텍처 구성 |
| **학습 & 평가** | GPU 기반 학습, 최종 정확도 0.8808 달성 |

#### 📌 Data Augmentation 설정
```python
image_generator = ImageDataGenerator(
    rotation_range=30,           # 랜덤 회전 ±30도
    brightness_range=[0.8, 1.0], # 밝기 조절
    zoom_range=0.3,              # 확대/축소
    width_shift_range=0.2,       # 좌우 이동
    height_shift_range=0.2,      # 상하 이동
    horizontal_flip=True,        # 좌우 반전
    vertical_flip=False          # 상하 반전 비활성화 (실제감 유지)
)
```

#### 📌 10개 클래스 구성
```
0: airplane  1: automobile  2: bird   3: cat   4: deer
5: dog       6: frog        7: horse  8: ship  9: truck
```

---

### 📌 2. `lastproject.ipynb` — TensorFlow Hub 전이학습 실험

> **TensorFlow Hub의 사전학습 MobileNetV2를 활용한 전이학습 실험**

#### 주요 내용

| 항목 | 내용 |
|------|------|
| **모델** | MobileNetV2 (TensorFlow Hub) |
| **입력 크기** | 224 × 224 × 3 (RGB) |
| **분류 클래스** | ImageNet 1,001개 클래스 |
| **실험 방식** | 꽃 사진 데이터셋(3,670장)으로 파인튜닝 |

#### 핵심 코드 흐름
```python
# 1. TF Hub에서 MobileNetV2 로드
classifier_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"
classifier = tf.keras.Sequential([hub.KerasLayer(classifier_url, input_shape=(224, 224, 3))])

# 2. 이미지 전처리 및 예측
result = classifier.predict(image[np.newaxis, ...])
predicted_class = np.argmax(result[0], axis=-1)

# 3. 라벨 매핑
imagenet_labels = np.array(open(labels_path).read().splitlines())
predicted_class_name = imagenet_labels[predicted_class]
```

---

### 📌 3. `alexnet.ipynb` — AlexNet 직접 구현

> **2012년 ImageNet 대회 우승 모델 AlexNet을 TensorFlow/Keras로 직접 구현**

#### AlexNet 아키텍처 구조
```
Input(224×224×3)
  → Conv2D(96, 11×11, stride=4) + ReLU + MaxPool
  → Conv2D(256, 5×5) + ReLU + MaxPool
  → Conv2D(384, 3×3) + ReLU
  → Conv2D(384, 3×3) + ReLU
  → Conv2D(256, 3×3) + ReLU + MaxPool
  → Flatten
  → Dense(4096) + ReLU + Dropout
  → Dense(4096) + ReLU + Dropout
  → Dense(1000) + Softmax
```

---

### 📌 4. `vggnet.ipynb` — VGGNet 구현

> **2014년 준우승 모델 VGGNet(Oxford Visual Geometry Group) 구현**

#### VGGNet 특징
- 3×3 소형 필터를 깊게 쌓는 단순하고 균일한 구조
- VGG-16 (13개 Conv + 3개 FC 레이어)
- 모든 Conv 레이어에 동일한 패딩/스트라이드 적용

---

### 📌 5. `Sigmoid_vs_ReLU_vs_Tanh.ipynb` — 활성화 함수 비교

> **세 가지 활성화 함수(Sigmoid, ReLU, Tanh)의 성능을 실험적으로 비교**

| 활성화 함수 | 특징 | 문제점 |
|------------|------|--------|
| **Sigmoid** | 출력 범위 (0,1) | 기울기 소실(Vanishing Gradient) |
| **Tanh** | 출력 범위 (-1,1) | 기울기 소실 (Sigmoid보다 개선) |
| **ReLU** | 양수 구간 선형 | 죽은 뉴런(Dead Neuron) 가능성 |

---

### 📌 6. `GoogLeNetStudy.ipynb` — GoogLeNet(Inception) 학습

> **2014년 ImageNet 대회 우승 모델 GoogLeNet의 핵심 구조인 Inception Module 학습**

#### Inception Module 특징
- 1×1, 3×3, 5×5 Conv를 병렬로 계산 후 채널 방향으로 합성
- 연산량 감소를 위한 1×1 bottleneck 레이어 활용
- 22개 레이어 깊이에도 파라미터 효율적

---

## 📊 모델 성능 비교

| 모델 | 등장 연도 | 특징 | ImageNet Top-5 오류율 |
|------|----------|------|---------------------|
| AlexNet | 2012 | 최초 딥러닝 우승 | 15.3% |
| GoogLeNet | 2014 | Inception Module | 6.7% |
| VGGNet | 2014 | 단순하고 깊은 구조 | 7.3% |
| ResNet | 2015 | Skip Connection | 3.57% |
| MobileNet | 2017 | 경량화, 모바일 최적화 | 29.4% (Top-1) |
| EfficientNet | 2019 | 복합 스케일링 | 2.9% (Top-1) |

---

## 📋 데이터 설명

데이터셋
- 이미지는 다양한 크기, RGB 색상 공간으로 표현, 1,000개의 클래스를 가지고 있음
- 이미지 클래스는 1~1,000까지 라벨로 되어 있음
- 50,000장의 이미지가 제공

Amazon Mechanical Turk 서비스의 이미지를 사람이 1,000개의 클래스로 분류
중복 이미지는 제거되어 있으며 120만개의 이미지는 1,000개의 클래스로 분류됨
클래스 정보는 WordNet 계층 구조를 따라 여러 분류로 나뉘어 있음
라벨은 제공되는 txt파일이며 txt파일의 row 번호가 이미지 번호임
- ILSVRC2010_ground_truth.txt 파일의 xx-번째 줄의 값이 77일 경우
  ILSVRC2010_val_000000xx.jpeg에 대응되는 클래스는 77임

## 🏗️ 모델 구축 가이드
1. 주어진 데이터셋의 이용 목적은 : binary vs. multi-class
2. 데이터의 이해 (중요)
3. 전처리, 특징 선택, 데이터 축소 등 (중요)
4. 모델은 3개 이상이며 비교 분석이 수반
5. 모델 평가 지표 선정
6. 훈련과 테스트 평가와 분석

## 🏆 결과 12월 23일 프로젝트 최우수상 

![KakaoTalk_20221224_123614680](https://user-images.githubusercontent.com/97416996/210133751-e3d601d9-bebe-4dd8-ba35-39094ce5887b.jpg)


> 이 프로젝트는 다양한 모델을 실험하고, 그 결과를 수치와 시각화로 증명한 경험이었어요.  
> 모델 성능을 분석하며 팀과 함께 전략을 세워 나간 덕분에, 발표 평가에서 최우수상을 받을 수 있었어요.  
> 이후 어떤 프로젝트에서도 '근거 있는 결과 도출'에 대한 감각을 갖게 된 값진 경험이었습니다.

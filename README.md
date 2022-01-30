# Vgg16-btc-predict

데이터셋은 [해당 오픈소스](https://github.com/philipperemy/deep-learning-bitcoin)를 사용해 만들었습니다. 5분봉 이미지를 vgg16 네트워크에 학습시켜 다음 5분봉의 상승과 하락을 추론합니다.
<img src="https://camo.githubusercontent.com/155250796f3ce74ec444d05d61e856739c35fcc0c46ecb4ea69f3b0eeb0a08c3/68747470733a2f2f626974636f696e2e6f72672f696d672f69636f6e732f6f70656e67726170682e706e67" width="30%" height="30%">

## How to get started
~~~~bash
git clone https://github.com/olivyou2/vgg16-btc-predict.git

# 데이터셋을 patterns 폴더에 넣어주세요
python train.py
~~~~

## 신경망 구성
vgg16 에 dense(2048) dense(1024) dense(2) 를 추가했습니다.
~~~~python
input_tensor = keras.Input(shape=(224, 224, 3), dtype='float32', name='input')

x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(input_tensor)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(x)
x = layers.MaxPooling2D((2,2))(x)

x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(x)
x = layers.MaxPooling2D((2,2))(x)

x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(x)
x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(x)
x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(x)
x = layers.MaxPooling2D((2,2))(x)

x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(x)
x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(x)
x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(x)
x = layers.MaxPooling2D((2,2))(x)

x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(x)
x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(x)
x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(x)
x = layers.MaxPooling2D((2,2))(x)

x = layers.Flatten()(x)
x = layers.Dense(4096, kernel_initializer='he_normal')(x)
x = layers.Dense(2048, kernel_initializer='he_normal')(x)
x = layers.Dense(1024, kernel_initializer='he_normal')(x)
output_tensor = layers.Dense(2, activation='softmax')(x)
~~~~

## Result

정확도 50퍼센트 ㅠ ㅠ loss 는 0.6940에서 과소적합 ㅠㅠ

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt


# ================= 1. 加载 MNIST 纯数字数据 =================
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # 归一化并增加通道维度 (N, 28, 28, 1)
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    return x_train, y_train, x_test, y_test


# ================= 2. 构建纯数字残差模型 =================
def build_digit_resnet():
    inputs = tf.keras.Input(shape=(28, 28, 1))

    # 初始特征提取
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)

    # 残差块：专门针对数字笔画（如 0 和 8 的圆圈，1 和 7 的折线）
    shortcut = layers.Conv2D(64, (1, 1), strides=2, padding='same')(x)

    x = layers.Conv2D(64, (3, 3), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # 输出层：只有 10 个类别 (0-9)
    outputs = layers.Dense(10, activation='softmax')(x)

    return models.Model(inputs, outputs)


# ================= 3. 主训练流程 =================
def train_digits():
    x_train, y_train, x_test, y_test = load_data()

    # 数据增强：解决连笔、歪斜、拍照模糊的关键
    datagen = ImageDataGenerator(
        rotation_range=10,  # 数字旋转一般不会太大
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )

    model = build_digit_resnet()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("启动纯数字模型训练...")
    history = model.fit(datagen.flow(x_train, y_train, batch_size=128),
                        epochs=15,  # 纯数字收敛很快
                        validation_data=(x_test, y_test))

    # 绘制结果图
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.title('Digit Accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Loss')
    plt.title('Digit Loss')
    plt.savefig('digit_training_result.png')

    # 导出 TFLite (开启量化压缩)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open('digit_recognizer.tflite', 'wb') as f:
        f.write(tflite_model)
    print("纯数字 TFLite 模型已生成！")


if __name__ == "__main__":
    train_digits()
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import gzip
import matplotlib.pyplot as plt
import os

# ================= 配置部分 =================
FILES = {
    'train_img': 'emnist-letters-train-images-idx3-ubyte.gz',
    'train_lbl': 'emnist-letters-train-labels-idx1-ubyte.gz',
    'test_img': 'emnist-letters-test-images-idx3-ubyte.gz',
    'test_lbl': 'emnist-letters-test-labels-idx1-ubyte.gz'
}


def load_gz_data(images_path, labels_path):
    """ 解析 EMNIST idx-ubyte.gz 格式文件 """
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16)
    # EMNIST 默认是 (N, 28, 28)，但需要转置矫正旋转问题
    images = images.reshape(len(labels), 28, 28)
    return images, labels


# ================= 模型构建 =================
def build_residual_model():
    """
    构建轻量级残差模型
    相比简单CNN，残差连接能更好地保留字母的笔画细节
    """
    inputs = tf.keras.Input(shape=(28, 28, 1))

    # 第一层：基础特征提取
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # 第二层：残差块 (Residual Block)
    # 使用 1x1 卷积匹配维度
    shortcut = layers.Conv2D(64, (1, 1), strides=2, padding='same')(x)

    x = layers.Conv2D(64, (3, 3), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, shortcut])  # 残差相加
    x = layers.Activation('relu')(x)

    # 第三层：深层抽象
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # 全连接分类头
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)  # 强力随机失活，防止过拟合
    outputs = layers.Dense(26, activation='softmax')(x)

    return models.Model(inputs, outputs)


# ================= 主训练函数 =================
def train_professional():
    # 1. 数据准备
    print("正在加载并矫正数据...")
    x_train, y_train = load_gz_data(FILES['train_img'], FILES['train_lbl'])
    x_test, y_test = load_gz_data(FILES['test_img'], FILES['test_lbl'])

    # 矫正 EMNIST 旋转问题
    x_train = np.transpose(x_train, (0, 2, 1))
    x_test = np.transpose(x_test, (0, 2, 1))

    # 归一化与维度调整
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_train = y_train - 1  # 1-26 映射到 0-25
    y_test = y_test - 1

    # 2. 实时数据增强
    # 模拟手写中的旋转、位移、缩放
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        shear_range=0.1,
        fill_mode='constant',
        cval=0
    )

    # 3. 编译模型
    model = build_residual_model()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 4. 回调函数：动态调整学习率 & 早停
    callbacks = [
        # 如果连续3个周期 loss 不降，学习率减半
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        # 如果连续6个周期准确率不升，停止训练并保留最好的权重
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True)
    ]

    # 5. 开始训练
    print("启动增强训练流程...")
    batch_size = 128
    history=model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(x_train) // batch_size,
        epochs=40,  # 有增强可以多练几轮
        validation_data=(x_test, y_test),
        callbacks=callbacks
    )

    # 6. 【新增】绘制 Loss 和 Accuracy 图表
    plt.figure(figsize=(12, 4))

    # 绘制 Accuracy 曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Test Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # 绘制 Loss 曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()  # 弹出窗口显示图表
    plt.savefig('training_log.png') # 如果想保存图片可以取消此行注释


    # 6. 导出 TFLite
    print("正在生成 handwritten_letter_model1.0.tflite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # 启用量化优化（可选，减小模型体积）
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open('handwritten_letter_model1.0.tflite', 'wb') as f:
        f.write(tflite_model)
    print("训练完成！")


if __name__ == "__main__":
    train_professional()
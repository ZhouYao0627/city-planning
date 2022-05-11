# -*- coding: utf-8 -*-
'''
使用CAM 将测试结果可视化——————————单张图像
'''
import numpy as np
from keras.models import Model
from keras import activations
from keras import backend as K
import tensorflow as tf
from tensorflow.python.framework import ops
from PIL import Image
import cv2
import utils
from keras.models import Sequential
from keras.layers.core import Lambda


class DefaultConfig():
    model_name = 'fsnet_05'
    train_data_path = '.\\dataset\\train\\'
    val_data_path = '.\\dataset\\test\\'
    checkpoints = '.\\checkpoints\\'
    modelpath = '\\model\\'
    fine_tune_model = '\\model\\'
    normal_size = 64
    epochs = 100
    batch_size = 32
    channles = 3  # or 3 or 1
    lr = 0.001
    lr_reduce_patience = 20
    early_stop_patience = 50  # 提前终止训练的步长
    finetune = False
    monitor = 'val_loss'
    image_shape = (128, 128, 3)
    classNumber = 6  # see dataset/tri


if __name__ == '__main__':

    from keras.models import load_model
    from my_models import baseline, baseline_seblock

    img_shape = (128, 128, 3)
    mask_shape = (8, 8, 1)
    num_class = 6

    # 载入模型和权重
    config = DefaultConfig()
    baseline_model = baseline(config, [1, 2, 2, 2], scope='baseline')
    baseline_weight = 'E:/datasets/Rcam-plusMelangerTaile/8KLSBackWindow/trained_model/baseline/modelpath/baseline.h5'
    model = load_model(baseline_weight)
    print('------Baseline load weight done------')

    # 载入图像——图像处理
    img_path = 'E:/datasets/Rcam-plusMelangerTaile/8KLSBackWindow/picked_out/1/755_2019_9_27.bmp'
    img_bgr = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = img_bgr.shape[:2]
    if h != 128 or w != 128:
        img_bgr = cv2.resize(img_bgr, (128, 128), interpolation=cv2.INTER_LINEAR)
    resnetv1_inputs = np.array([img_bgr])

    pred = model.predict(resnetv1_inputs)
    class_idx = np.argmax(pred[0])

    class_output = model.output[:, class_idx]

    # 需根据自己情况修改2. 把block5_conv3改成自己模型最后一层卷积层的名字
    last_conv_layer = model.get_layer("conv2d_18")

    grads = K.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([resnetv1_inputs])

    ##需根据自己情况修改3. 512是我最后一层卷基层的通道数，根据自己情况修改
    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)  # relu激活。
    heatmap /= np.max(heatmap)
    #
    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_NEAREST)
    # img = img_to_array(image)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)  # 将cam图像叠加到原图上
    cv2.imwrite('image_grad_cam.png', superimposed_img)
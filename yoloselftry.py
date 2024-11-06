import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Input
from tensorflow.keras.layers import MaxPooling2D, Reshape, Concatenate



#MaxPooling2D() 預設是2*2

#好像沒說過padding = "same"是什麼 這個堆疊會讓輸入輸出維持同大小
def yolo_model():
    input_layer = Input(shape= (640,640,3))
    #第一層卷積
    x = Conv2D(32, 3, padding='same')(input_layer)# 640 x 640 x 32
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D()(x) # 320 x 320 x 32  640 -> ((640 - 2) / 2) + 1 = 320

    #二
    x = Conv2D(64, 3, padding='same')(x)# 320 x 320 x 64
    x = BatchNormalization()(x) 
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D()(x) # 160 x 160 x 64  320 -> ((320 - 2) / 2) + 1 = 160
    #三
    x = Conv2D(128, 3, padding='same')(x) # 160 x 160 x 128
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D()(x)# 80 x 80 x 128  160 -> ((160 - 2) / 2) + 1 = 80
    #四
    x = Conv2D(256, 3, padding='same')(x) # 80 x 80 x 256
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D()(x)# 40 x 40 x 256  80 -> ((80 - 2) / 2) + 1 = 40
    
    #五
    x = Conv2D(512, 3, padding='same')(x)# 40 x 40 x 512
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    
    # 輸出層: 每個grid cell預測5個值 (x, y, w, h, confidence)
    output = Conv2D(5, 1, padding='same')(x)# 40 x 40 x 5
    
    return Model(input_layer, output)



'''
y_pred 和 y_true 的內容都是 [batch_size, 40, 40, 5]

[..., 0:2] 取最後一維的前兩個值（x,y坐標）
[batch_size, 40, 40, 2]
[..., 2:4] 取最後一維的第3、4個值（寬度和高度）
[batch_size, 40, 40, 2]
# [..., 4:5] 取最後一維的第5個值（置信度）
[batch_size, 40, 40, 1]
'''
def loss(y_true, y_pred):
    """YOLO損失函數的實現"""
    # 解包預測值
    pred_xy = y_pred[..., 0:2]
    pred_wh = y_pred[..., 2:4]
    pred_conf = y_pred[..., 4:5]
    
    # 解包真實值
    true_xy = y_true[..., 0:2]
    true_wh = y_true[..., 2:4]
    true_conf = y_true[..., 4:5]
    
    # 座標損失
    xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy))
    # 1. true_xy - pred_xy：計算預測座標和真實座標的差異
    # 2. tf.square()：計算差異的平方（均方誤差）
    # 3. tf.reduce_sum()：將所有誤差加總
    wh_loss = tf.reduce_sum(tf.square(true_wh - pred_wh))
    # 1. true_wh - pred_wh：計算預測寬高和真實寬高的差異
    # 2. tf.square()：計算差異的平方
    # 3. tf.reduce_sum()：將所有誤差加總
    # 置信度損失
    conf_loss = tf.reduce_sum(tf.square(true_conf - pred_conf))
    # 1. true_conf - pred_conf：計算預測置信度和真實置信度的差異
    # 2. tf.square()：計算差異的平方
    # 3. tf.reduce_sum()：將所有誤差加總
    return xy_loss + wh_loss + conf_loss
    # 將三個損失簡單相加作為總損失




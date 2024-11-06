import tensorflow as tf
from tensorflow import keras
import numpy as np

#數字決定層的寬度
#Dense 全連接層 所有神經元與上層相連
'''
activation: 激活函數

relu: 對負值輸出0,正值保持不變
softmax: 常用於最後一層,將輸出轉換為概率分布
sigmoid: 輸出範圍在0-1之間
tanh: 輸出範圍在-1到1之間
input_shape 定義數據輸入的形狀

'''
model = {
    keras.layer.Dense(64,activation = "relu", input_shape = (784,)),
    keras.layer.Dense(32,activation = "relu"),
    keras.layer.Dense(10,activation = "softmax"),
}

# 不同類型數據的 input_shape 示例
models = {
    # 向量數據
    'vector': keras.Sequential([
        keras.layers.Dense(64, input_shape=(100,))  # 100個特徵的向量 一維數據
    ]),
    
    # 灰度圖片
    'grayscale_image': keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1)) #二維數據 單通道
    ]),
    
    # 彩色圖片
    'color_image': keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), input_shape=(224, 224, 3)) #二維數據 三通道
    ]),
    
    # 時序數據
    'time_series': keras.Sequential([
        keras.layers.LSTM(64, input_shape=(50, 10))  # 50個時間步，每步10個特徵 
    ])
}
'''
動態輸入
使用None來表示可變的維度
'''

model = keras.Sequential([
    # 第一個維度可變，第二個維度必須是100
    keras.layers.Dense(64, input_shape=(None, 100)),
    
    # 圖片大小可變，但必須是3通道
    keras.layers.Conv2D(32, (3, 3), input_shape=(None, None, 3))
])


'''
模型編譯
optimizer 優化器 決定模型如何更新權重
adam: 自適應學習率
sgd: 隨機梯度下降
rmsprop:自適應學習率的優化版 適合RNN 收斂速度快 但需要更多內存

關於rmsprop:
他會累積平方梯度並且進行權重更新
但建議學習率要調低
並且衰減率(rho)需要調整較高
可以配合梯度裁減使用
optimizer = keras.optimizers.RMSprop(clipvalue=1.0)
不然你的梯度會一直爆炸

loss 損失函數
分類問題: categorical_crossentropy 或 sparse_categorical_crossentropy
回歸問題: mse(均方誤差) 或 mae(平均絕對誤差)

metrics 評估指標
如準確率、精確率等
'''

model.compile(
    optimizer = "adam",
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


'''
dropout 層
regularization技術 (正規化)
通常建議越深的層關閉越多
'''

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),  # 50% 的神經元會被隨機關閉
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),  # 30% 的神經元會被隨機關閉
    keras.layers.Dense(10, activation='softmax')
])

'''
CNN裡面的dropout

'''
model = keras.Sequential([
    # 卷積層通常使用較小的 dropout 率
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.Dropout(0.1),
    
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Dropout(0.2),
    
    keras.layers.Flatten(),
    # 全連接層使用較大的 dropout 率
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    
    keras.layers.Dense(10, activation='softmax')
])

'''
RNN裡面的dropout

'''
model = keras.Sequential([
    # 對於 LSTM，可以分別對輸入和循環連接使用 dropout
    keras.layers.LSTM(64, 
                     dropout=0.2,          # 輸入的 dropout 率
                     recurrent_dropout=0.2  # 循環連接的 dropout 率
                    ),
    keras.layers.Dense(10, activation='softmax')
])

'''
建議訓練時使用dropout
預測時記得關閉 雖然tensorflow有設定好

'''


model.fit(
    x_train, 
    y_train,
    epochs=10,
    validation_data=(x_val, y_val)
)  # 訓練時 Dropout 生效

# 預測時自動關閉 Dropout
predictions = model.predict(x_test)  # 預測時 Dropout 關閉


'''
常見的dropout使用
'''

'''
甚麼是歸一化?
可以用於加速訓練並且提高模型穩定性的技術
因為在訓練時 通常會將數據拆為多個batch進行訓練
所以會利用統計分析處理數據分布的偏移以及縮放
利用移動平均來追蹤整個數據集

優點:
它讓我們可以使用更大的學習率
減少對初始化的依賴

original_data = [1, 5, 10, 15, 20]  # 均值=10, 方差=50

# 經過批量歸一化後
normalized_data = [-1.26, -0.63, 0, 0.63, 1.26]  # 均值≈0, 方差≈1

他讓數據範圍被規範
並且減少梯度消失或炸掉的狀況
加速神經網路收斂
並且提供一點點正則化效果
所以通常我會配合dropout使用
數學實現:

'''
def __init__(self, epsilon=1e-5, momentum=0.99):
    self.epsilon = epsilon        # 防止除以零的小數值
    self.momentum = momentum      # 移動平均的權重
    self.running_mean = None      # 儲存整體數據的平均值
    self.running_var = None       # 儲存整體數據的方差
    if training:
        #計算當前batch的平均值和方差
        batch_mean = np.mean(x, axis=0)  # 對每個特徵計算平均值
        batch_var = np.var(x, axis=0)    # 對每個特徵計算方差
        
        #更新當前整體的統計量（使用移動平均）
        if self.running_mean is None:  # first batch
            self.running_mean = batch_mean
            self.running_var = batch_var
        else:  # 後續批次
            # 使用移動平均更新
            self.running_mean = self.momentum * self.running_mean + \
                            (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + \
                            (1 - self.momentum) * batch_var
        
        #對當前的batch進行標準化(注意 是以新的batch加上已經保存的整體進行標準化)
        x_normalized = (x - batch_mean) / np.sqrt(batch_var + self.epsilon)
    



'''
常見的dropout使用
'''
# 1. 配合使用批量歸一化（Batch Normalization） 通常建議  連接層>BN層>激活函式
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    
    keras.layers.Dense(64, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    
    keras.layers.Dense(10, activation='softmax')
])

# 2. 使用較大的網絡
model_with_dropout = keras.Sequential([
    keras.layers.Dense(1024, activation='relu'),  # 較大的網絡
    keras.layers.Dropout(0.5),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])



'''
CNN的邏輯解析
以64 32 10的結構為例

輸出層設定(28,28,1)-->第一組卷積層[64個3*3的過濾器 26*26*64-->批量歸一化-->ReLU激活-->最大池化 (2*2) 13*13*64]-->第二組卷積層[32個3*3過濾器 11*11*32-->批量歸一化-->ReLU激活-->最大池化2*2 5*5*32]-->展平層(800維度)-->全連接層 128神經元-->批量歸一化-->ReLU激活-->dropout0.5-->輸出層10神經元 softmax

過濾器(卷積核)
我們可以透過padding模擬更大的過濾器 節省資源 兩個3*3過濾器可以模擬出一個5*5的過濾器
而3*3就是可以檢測到的特徵範圍

步長 stride
卷積或池化操作時，過濾器或池化窗口在輸入上移動的距離
輸入數據：
[1 2 3 4]
[5 6 7 8]
[9 0 1 2]

3x3 過濾器移動（*表示當前位置） 步長為1：
第一步：        第二步：
[* * *]        [1 * * *]
[* * *] 2      [5 * * *] 2
[* * *]        [9 * * *]

輸入數據：
[1 2 3 4]
[5 6 7 8]
[9 0 1 2]

3x3 過濾器移動（*表示當前位置） 步長為2：
第一步：        第二步（直接跳2格）：
[* * *]        [1 2 * * *]
[* * *] 2      [5 6 * * *] 2
[* * *]        [9 0 * * *]
可以用來降低特徵圖的大小
大步長減少計算量
輸入 28x28，步長=1 vs 步長=2

步長=1：
輸出大小 26x26 = 676 個點需要計算

步長=2：
輸出大小 13x13 = 169 個點需要計算
建議淺層使用小步長保留詳細特徵訊息
深層可以用大步長減少計算量

維度的計算 從28*28*1到26*26*64
輸入大小 28*28
過濾器大小 3*3
步長(stride) 1
padding 'valid'（不填充）
# 新維度 = (原維度 - 過濾器大小 + 1) / 步長
# (28 - 3 + 1) = 26

# 64是過濾器的數量，決定了輸出通道數

甚麼是最大池化層
最大池化層 (MaxPooling) 是 CNN 中用於降維的一種技術
工作原理
# 假設我們有一個 4x4 的特徵圖
feature_map = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

# 使用 2x2 的最大池化，結果會是：
pooled = np.array([
    [6, 8],    # max([1,2,5,6]), max([3,4,7,8])
    [14, 16]   # max([9,10,13,14]), max([11,12,15,16])
])
不同類型的池化層
# 最大池化（保留最大值）
keras.layers.MaxPooling2D(pool_size=(2, 2))

# 平均池化（計算平均值）
keras.layers.AveragePooling2D(pool_size=(2, 2))

# 全局池化（整個特徵圖變成一個值）
keras.layers.GlobalMaxPooling2D()
keras.layers.GlobalAveragePooling2D()
參數設定
pool_layer = keras.layers.MaxPooling2D(
    pool_size=(2, 2),    # 池化窗口大小
    strides=None,        # 步長，默認等於 pool_size
    padding='valid'      # 填充方式
)
池化層可以減少參數量
# 原始特徵圖：28x28x32 = 25,088 個參數
# 池化後：14x14x32 = 6,272 個參數
# 減少了 75% 的參數量
也可以提供平移不變性

# 原始圖像中的特徵位置稍有偏移
feature1 = [
    [0, 0, 7, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
]

feature2 = [
    [0, 7, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
]

# 經過 2x2 最大池化後，這兩種情況的輸出相同
pooled = [
    [7, 0],
    [0, 0]
]
# 使用 2×2 的池化窗口，步長為 2
# 新維度 = 原維度 / 池化窗口大小
# 26 ÷ 2 = 13
所以變成13*13*64
# 通道數（64）保持不變
通常池化層會用2*2
因為3*3會損失太多訊息

後續維度計算自己推 我懶


'''
# 3x3 過濾器的示例
filter_example = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])  # 這是一個垂直邊緣檢測過濾器

# 假設輸入圖像片段為
image_patch = np.array([
    [100, 150, 200],
    [105, 155, 205],
    [110, 160, 210]
])

# 卷積操作就是兩個矩陣的元素對應相乘再相加
result = np.sum(filter_example * image_patch)

'''
典型的CNN架構
'''
model = keras.Sequential([
    # 第一組卷積+池化
    keras.layers.Conv2D(64, (3, 3),padding='same', input_shape=(28, 28, 1)),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),  # 28x28 -> 14x14
    
    # 第二組卷積+池化
    keras.layers.Conv2D(32, (3, 3),padding='same'),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),  # 14x14 -> 7x7
    
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation='softmax')
])

'''
空洞卷積(dilated convolution)
'''
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np
import os
from IPython.display import HTML
import matplotlib.pyplot as plt
# %matplotlib inline
from matplotlib import rcParams
rcParams.update({'xtick.color':'w','ytick.color':'w','axes.labelcolor':'gold'})
import japanize_matplotlib


# ## ファイル読み込み

# + {"code_folding": [3, 5]}
def mk_dataframe(path):
    """
    pathに元ずいてdataframeを作る。
    path:str
        train or test/pos or neg
    files:list
        text data to read
    """
    data = []
    files = [x for x in os.listdir(path) if x.endswith('.txt') ]
    
    for text_name in files:
        # ファイルを読み込む
        with open(path+text_name,'r') as text_data:
            text = text_data.read()
        # IDとreview読み込み
        text_num = text_name.rstrip('.txt')
        ID,review = text_num.split('_')
        #  バイナリー値の代入
        if int(review) >= 7:
            label = "1"
        elif int(review) <= 4:
            label = "0"
        else:
            label = ""
        data.append([ID,review,label,text])
    df = pd.DataFrame(data,columns=['ID','review','label','text'],index=None)
    return df


# -

# それぞれのデータを読み込む
train_pos_df = mk_dataframe('../aclImdb/train/pos/')
train_neg_df = mk_dataframe('../aclImdb/train/neg/')
test_pos_df = mk_dataframe('../aclImdb/test/pos/')
test_neg_df = mk_dataframe('../aclImdb/test/neg/')


# + {"code_folding": []}
def shuffle_data(pos_data,neg_data):
    '''
    posとnegのdataframeを結合する
    '''
    full_df = pd.concat([pos_data,neg_data]).sample(frac=1,random_state=1)
    return full_df
    


# -

# 訓練用とテスト用データの作成
train_df = shuffle_data(train_pos_df,train_neg_df)
test_df = shuffle_data(test_pos_df,test_neg_df)
train_df.shape,test_df.shape
train_df.head(10)

# 文章のサンプル表示
HTML(train_df.text.iloc[0])

# ユニークな評価数 ラベル数
print('review:\n{0}\nlabel:\n{1}'.format(train_df.review.value_counts(),train_df.label.value_counts()))

plt.figure(figsize=(15, 10))
plt.hist([len(sample) for sample in list(train_df.text)], 50)
plt.xlabel('テキストの長さ')
plt.ylabel('テキストの量')
plt.title('テキストの分布',color='gold')
plt.show()

# ## 前処理(labelを使う場合)

# X,yにデータを分ける
train_data = train_df.iloc[:,2:]
train_X = train_df.iloc[:,3].values
train_y = train_df.iloc[:,2].values
print(train_y.shape)

# 単語のダミーの作成
from sklearn.feature_extraction.text import CountVectorizer
CountVector = CountVectorizer()
docs =  train_X
bag = CountVector.fit_transform(docs)
print(CountVector.vocabulary_)

# # ダミー化させた特徴量の抽出
train_X_features = bag.toarray()
print(train_X_features.shape)

# ボキャブラリーの全体像
vocab = CountVector.get_feature_names()
print(vocab)

# ボキャブラリーの数それぞれ
dist = np.sum(train_X_features,axis=0)
print(dist)

print("count:word")
for word,count in zip(vocab,dist):
    print("{0}:{1}".format(count,word))

# ## 機械学習モデル作成(labelを使う場合)

# +
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score,auc

X_train,X_test,y_train,y_test = train_test_split(train_X_features,train_y)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
accuracy_score = accuracy_score(y_test.astype('int'),y_pred.astype('int'))
# auc_score = auc(y_test.astype('int'),y_pred.astype('int'))
roc_auc_score = roc_auc_score(y_test.astype('int'),y_pred.astype('int'))
print("accuracy_スコア:{0}\nroc_aucスコア:{1}\n".format(accuracy_score,roc_auc_score))
# -

# ## 深層学習モデル作成(label使う場合)

# +
# tensorflowとkerasのバージョン確認
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import layers

print(tf.VERSION)
print(tf.keras.__version__)

# +
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train_X_features,train_y)

#深層学習モデル作成
DNN_model = tf.keras.Sequential()
# -

# ## Sequentialモデル

# ユニット数が64の全結合を2つ入れる
DNN_model.add(layers.Dense(64,activation='relu',input_shape=(74849,)))
DNN_model.add(layers.Dense(64,activation='relu'))
# 出力ユニット数が10のソフトマックス層を追加します：
# DNN_model.add(layers.Dense(10, activation='softmax'))
# 活性化関数の適用
layers.Dense(64, activation=tf.sigmoid)
# 正則化
layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))
layers.Dense(64,bias_regularizer=tf.keras.regularizers.l2(0.01))
# カーネルをランダム直交行列で初期化した全結合層：
layers.Dense(64, kernel_initializer='orthogonal')
# バイアスベクトルを2.0で初期化した全結合層：
layers.Dense(64, bias_initializer=tf.keras.initializers.constant(2.0))

# モデル作成
DNN_model.compile(optimizer=tf.train.AdamOptimizer(0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# DNN_model.evaluate(X_train.astype('float'),y_train.astype('float'))
DNN_model.fit(X_train,y_train,epochs=10,batch_size=30)

y_pred = DNN_model.predict(X_test,batch_size=5)
# ２値に整形する
score = DNN_model.evaluate(X_train,y_train)
print(score)
print(DNN_model.metrics_names)

# %precision
print(y_pred.shape)
mask = y_pred.astype('int') == 1
y_pred.astype('int')[mask]


# +
from sklearn.metrics import accuracy_score,roc_auc_score,auc

accuracy_score = accuracy_score(y_test.astype('int'),y_pred.astype('int'))
# auc_score = auc(y_test.astype('int'),y_pred.astype('int'))
roc_auc_score = roc_auc_score(y_test.astype('int'),y_pred.astype('int'))
print("accuracy_スコア:{0}\nroc_aucスコア:{1}\n".format(accuracy_score,roc_auc_score))


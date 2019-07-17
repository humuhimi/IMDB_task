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
train_pos_df = mk_dataframe('Downloads/aclImdb/train/pos/')
train_neg_df = mk_dataframe('Downloads/aclImdb/train/neg/')
test_pos_df = mk_dataframe('Downloads/aclImdb/test/pos/')
test_neg_df = mk_dataframe('Downloads/aclImdb/test/neg/')


# + {"code_folding": [0]}
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

train_data = train_df.iloc[:,2:]
train_X = train_df.iloc[:,3].values
train_y = train_df.iloc[:,2].values

from sklearn.feature_extraction.text import CountVectorizer
CountVector = CountVectorizer()
docs =  train_X
bag = CountVector.fit_transform(docs)
print(CountVector.vocabulary_)

print(bag.toarray())



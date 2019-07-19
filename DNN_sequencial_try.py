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

predictions = np.round(DNN_model.predict(X_test))
print(predictions[:100])


# +

pd.set_option('display.max_rows', 50)
predictions[:10]
# -

correct = y_test.[:,newaxis]
print(correct)

# +
from sklearn.metrics import accuracy_score,roc_auc_score,auc

accuracy_score = accuracy_score(y_test.astype('int'),y_pred.astype('int'))
# auc_score = auc(y_test.astype('int'),y_pred.astype('int'))
roc_auc_score = roc_auc_score(y_test.astype('int'),y_pred.astype('int'))
print("accuracy_スコア:{0}\nroc_aucスコア:{1}\n".format(accuracy_score,roc_auc_score))


import random

import os
import numpy
import pymysql
import sklearn
# gensim：从原始的非结构化文本中，无监督地学习到文本隐层地主题向量表达，主要用于主题建模、文档相似性处理
from gensim.models import Word2Vec
from keras import Input, Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Embedding, Dense, concatenate, Convolution1D, MaxPooling1D, Flatten, BatchNormalization, \
	Dropout, Multiply
import torch
from keras.optimizers import Adam
from keras.utils import np_utils
from keras_preprocessing.sequence import pad_sequences
# Tokenizer用于向量化文本，或将文本转换为序列，实际上只是生成了一个字典、统计了词频等信息，没有把文本转换成需要的向量表示
from keras_preprocessing.text import Tokenizer
import tensorflow as tf

from utils import evaluate, read_data

# from main2 import read_data

sg = 1
window = 7
min_count = 0
negative = 5
sample = 0.00025
hs = 1

MAX_SEQUENCE_LENGTH_s = 10
MAX_SEQUENCE_LENGTH_d = 35
EMBEDDING_DIM = 32  # 词向量的大小
epoch = 200
batch_size = 640

db = pymysql.connect(
	host="172.29.7.222",
	port=3306,
	user="root",
	password="1234",
	database="bugzilla"
)

"""
///////////////////////////////
//   连接数据库，读取、划分数据   //
///////////////////////////////
"""
mycursor = db.cursor()
# summary：text类型
# description：text类型
id, summary, description, resolution, text, preprocessed12, preprocessed12345, sentiment, preprocessed_summary, preprocessed_description = read_data(
	mycursor)
mycursor.close()
db.close()

# index = [i for i in range(len(summary))]
# random.shuffle(index)
# summary = summary[index]
# description = description[index]
# sentiment = sentiment[index]

# 划分train、test数据集
summary_train, summary_test, description_train, description_test, sentiment_train, sentiment_test, y_train, y_test = sklearn.model_selection.train_test_split(
	summary, description, sentiment, resolution, test_size=0.1, random_state=0)

"""
///////////////////////////////
//     为情绪分数构建tensor     //
///////////////////////////////
"""
# 将类别向量转换为二进制（0和1）
sentiment_train = np_utils.to_categorical(sentiment_train, num_classes=3)
sentiment_test = np_utils.to_categorical(sentiment_test, num_classes=3)
# 创建tensor
sentiment_train = torch.tensor(sentiment_train)
sentiment_test = torch.tensor(sentiment_test)
# 增加一个维度：在第1维度（下标从0开始）上增加"1"
# [1., 2.] => [[1.],
#               2.]]
sentiment_train = sentiment_train.unsqueeze(1)
sentiment_test = sentiment_test.unsqueeze(1)

print(sentiment_train.shape)

"""
//////////////////////////////////////////////////
//   构建word2vec模型，使用Tokenizer分词器进行分词   //
//////////////////////////////////////////////////
"""
# 使用word2vec模型，将文本转换为固定长度的k维数值向量
print("word2vec_model------")
W2V_MODEL_s = Word2Vec(summary_train, sg=1, size=EMBEDDING_DIM, window=7, min_count=0, negative=5, sample=0.00025,
                       hs=1)  # 构建词向量模型
W2V_MODEL_d = Word2Vec(description_train, sg=1, size=EMBEDDING_DIM, window=7, min_count=0, negative=5, sample=0.00025,
                       hs=1)

# 创建分词器
print("Tokenizer----------")
tokenizer_s = Tokenizer()
tokenizer_s.fit_on_texts(summary)  # 分词器方法：实现分词
word_index_s = tokenizer_s.word_index  # 获取单词索引
print("Pad_sequence----------")

summary_train = tokenizer_s.texts_to_sequences(summary_train)  # 将一个句子拆分成单词构成的列表（单词索引序列）
summary_train = pad_sequences(summary_train, maxlen=MAX_SEQUENCE_LENGTH_s)  # keras只能接受长度相同的序列输入，需要使用填充来将序列转化为一个长度相同的新序列
summary_test = tokenizer_s.texts_to_sequences(summary_test)
summary_test = pad_sequences(summary_test, maxlen=MAX_SEQUENCE_LENGTH_s)

tokenizer_d = Tokenizer()
tokenizer_d.fit_on_texts(description)
word_index_d = tokenizer_d.word_index
description_train = tokenizer_d.texts_to_sequences(description_train)
description_train = pad_sequences(description_train, maxlen=MAX_SEQUENCE_LENGTH_d)
description_test = tokenizer_d.texts_to_sequences(description_test)
description_test = pad_sequences(description_test, maxlen=MAX_SEQUENCE_LENGTH_d)

# sentiment_train = numpy.array(sentiment_train)
# sentiment_test = numpy.array(sentiment_test)


# sentiment_train = sentiment_train.reshape((sentiment_train.shape[0], sentiment_train.shape[1],1))
# sentiment_test = sentiment_test.reshape((sentiment_test.shape[0], sentiment_test.shape[1], 1))
# y_train = np_utils.to_categorical(y_train, num_classes=7)
# y_test = np_utils.to_categorical(y_test, num_classes=7)

"""
//////////////////////////////
//        获取嵌入矩阵        //
//////////////////////////////
"""
print('Embedding----------')
nb_words_s = len(word_index_s)  # 所有分词的数量
embedding_matrix_s = numpy.zeros((nb_words_s + 1, EMBEDDING_DIM))  # 根据Tokenizer分词的数量来构建嵌入矩阵

nb_words_d = len(word_index_d)
embedding_matrix_d = numpy.zeros((nb_words_d + 1, EMBEDDING_DIM))

# 获取所有单词的词向量，并使用词向量构建embedding_matrix矩阵
for word, i in word_index_s.items():
	try:
		embedding_vector_s = W2V_MODEL_s.wv[word]  # 获取Tokenizer分词得到的某个word的词向量表示
	except KeyError:
		continue
	if embedding_vector_s is not None:
		embedding_matrix_s[i] = embedding_vector_s

for word, i in word_index_d.items():
	try:
		embedding_vector_d = W2V_MODEL_d.wv[word]
	except KeyError:
		continue
	if embedding_vector_d is not None:
		embedding_matrix_d[i] = embedding_vector_d

"""
//////////////////////////////
//         构建嵌入层         //
//////////////////////////////
"""
# embedding层
embedding_layer_s = Embedding(nb_words_s + 1, EMBEDDING_DIM, weights=[embedding_matrix_s],
                              input_length=MAX_SEQUENCE_LENGTH_s, trainable=False)
embedding_layer_d = Embedding(nb_words_d + 1, EMBEDDING_DIM, weights=[embedding_matrix_d],
                              input_length=MAX_SEQUENCE_LENGTH_d, trainable=False)

"""
//////////////////////////////
//         构建输入层         //
//////////////////////////////
"""
inputs_s = Input(shape=(MAX_SEQUENCE_LENGTH_s,), name='inputs_s')  # summary输入层
inputs_d = Input(shape=(MAX_SEQUENCE_LENGTH_d,), name='input_d')  # description输入层
senti_input = Input(shape=(sentiment_train.shape[1], sentiment_train.shape[2],),
                    name='senti_input')  # sentiment score输入层

sentence_input_s = embedding_layer_s(inputs_s)
sentence_input_d = embedding_layer_d(inputs_d)

"""
//////////////////////////////
//         构建卷积层         //
//////////////////////////////
"""
# s、d、senti分别传入不同的卷积层
conv1D_s = (Convolution1D(64, 1, padding='same', activation='relu'))(sentence_input_s)
conv1D_d = (Convolution1D(64, 1, padding='same', activation='relu'))(sentence_input_d)
conv1D_senti = (Convolution1D(64, 1, padding='same', activation='relu'))(senti_input)
# 三个卷积层分别进行BatchNormalization
conv1D_s = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                              gamma_initializer='ones', moving_mean_initializer='zeros',
                              moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                              beta_constraint=None, gamma_constraint=None)(conv1D_s)
conv1D_d = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                              gamma_initializer='ones', moving_mean_initializer='zeros',
                              moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                              beta_constraint=None, gamma_constraint=None)(conv1D_d)
conv1D_senti = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                  beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
                                  moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                  beta_constraint=None, gamma_constraint=None)(conv1D_senti)

"""
//////////////////////////////
//       Concatenate       //
//////////////////////////////
"""
# 多维输入一维化，并进行concatenate
conv1D_s = Flatten()(conv1D_s)
conv1D_d = Flatten()(conv1D_d)
conv1D_senti = Flatten()(conv1D_senti)
concatenation = concatenate([conv1D_s, conv1D_d, conv1D_senti])

"""
//////////////////////////////
//         Attention        //
//////////////////////////////
"""
# Dense层：全连接神经网络层
attention_probs = Dense(2944, activation='softmax', name='attention_vec')(concatenation)
attention_mul = Multiply()([concatenation, attention_probs])  # 相当于两个矩阵相乘
# flatten = Flatten()(concatenation)
# attention_layer = tf.keras.layers.Attention()(concatenation)

"""
//////////////////////////////
//          Dropout         //
//////////////////////////////
"""
dropout = Dropout(0.5, noise_shape=None, seed=None)(attention_mul)
"""
//////////////////////////////
//      Fully Connected     //
//////////////////////////////
"""
# 参数：该层的神经元数量，该层使用的激活函数
predictions = Dense(1, activation='sigmoid')(dropout)

# sgd = Adam(lr=0.01)
model = Model(inputs=[inputs_s, inputs_d, senti_input], outputs=predictions)
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
# early_stopping = EarlyStopping(monitor='acc', patience=3)
reduceLROnPlateau = ReduceLROnPlateau(monitor='acc', factor=0.1, patience=3, mode='auto',
                                      min_delta=0.001, cooldown=0, min_lr=0)
# history = model.fit([summary_train, description_train, sentiment_train], y_train, batch_size=batch_size, epochs=epoch, verbose=1, callbacks=[early_stopping, reduceLROnPlateau])
history = model.fit([summary_train, description_train, sentiment_train], y_train, batch_size=batch_size, epochs=epoch,
                    verbose=1, callbacks=[reduceLROnPlateau])
score = model.evaluate([summary_test, description_test, sentiment_test], y_test, verbose=0)
y_predict = model.predict([summary_test, description_test, sentiment_test])

for index, value in enumerate(y_predict):
	if (value >= 0.5):
		y_predict[index] = 1
	elif (value < 0.5):
		y_predict[index] = 0

# print("y_test", y_test)
# print("y_predict", y_predict)

precision, recall, f = evaluate(y_test, y_predict)
print("score:", score)

with open(os.path.join('/home/chenheng/ch/baseline_copy/CNN_output/Arshad_output.txt'), 'w') as output:
	output.write("precision: " + str(precision) + "\n")
	output.write("recall: " + str(recall) + "\n")
	output.write("f: " + str(f) + "\n")
	output.write("score: " + str(score) + "\n")

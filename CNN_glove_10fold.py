import numpy as np
import pymysql
from gensim.models import Word2Vec
from keras import Input, Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Embedding, Dense, concatenate, Conv1D, MaxPooling1D, Flatten, BatchNormalization, \
    Dropout, Multiply
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import torch
from keras.optimizers import Adam
from keras.utils import np_utils
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import random
from utils import evaluate
import os
from sklearn import metrics
from keras import backend as K

def read_data(fold_id, FLAG):
    summary_train, summary_test, descrip_train, descrip_test, y_train, y_test, sentiment_train, sentiment_test, creator_train, creator_test, role_train, role_test, freq_train, freq_test = [], [], [], [], [], [], [], [], [], [], [], [], [], []
    dict = {"FIXED": 1, "INVALID": 0, "DUPLICATE": 0, "WONTFIX": 0, "INCOMPLETE": 0, "WORKSFORME": 0, "EXPIRED": 0,
            "MOVED": 0, "INACTIVE": 0}
    dict_senti = {"negative": 0, "positive": 1, "neutral": 2}

    db = pymysql.connect(
        host="172.29.7.222",
        port=3306,
        user="root",
        password="1234",
        database="bugzilla"
    )
    mycursor = db.cursor()
    if FLAG == 0:  # 随机平均划分十折
        # 读取测试集数据
        mycursor.execute(
            "SELECT preprocessed_summary, preprocessed_description, resolution, sentiment, creator, creator_developer, creator_freq from enhancement where folds_id=" + str(fold_id))
        myresult = mycursor.fetchall()
        result = myresult[:]

        for index, element in enumerate(result):
            summary_test.append(element[0].replace("\n", " "))
            descrip_test.append(element[1].replace("\n", " "))
            y_test.append(dict[element[2]])
            sentiment_test.append(dict_senti[element[3]])
            if element[5] is None:
                role_test.append(0)
            else:
                role_test.append(element[5])
            creator_test.append(element[4])
            freq_test.append(int(element[6]))

        # 读取训练集数据
        mycursor.execute(
            "SELECT preprocessed_summary, preprocessed_description, resolution, sentiment, creator, creator_developer, creator_freq from enhancement where folds_id is not null and folds_id!=" + str(
                fold_id))
        myresult = mycursor.fetchall()
        result = myresult[:]

        for index, element in enumerate(result):
            summary_train.append(element[0].replace("\n", " "))
            descrip_train.append(element[1].replace("\n", " "))
            y_train.append(dict[element[2]])
            sentiment_train.append(dict_senti[element[3]])
            if element[5] is None:
                role_train.append(0)
            else:
                role_train.append(element[5])
            creator_train.append(element[4])
            freq_train.append(element[6])
    if FLAG == 1:  # 十个项目划分十折
        product_list = ["Bugzilla", "SeaMonkey", "Core Graveyard", "Core", "MailNews Core", "Toolkit", "Firefox",
                        "Thunderbird", "Calendar", "Camino Graveyard"]
        # 读取测试集数据
        mycursor.execute(
            "SELECT preprocessed_summary, preprocessed_description, resolution, sentiment, creator, creator_developer, creator_freq from enhancement where sentiment is not null and product='" +
            product_list[fold_id] + "'")
        myresult = mycursor.fetchall()
        result = myresult[:]

        for index, element in enumerate(result):
            summary_test.append(element[0].replace("\n", " "))
            descrip_test.append(element[1].replace("\n", " "))
            y_test.append(dict[element[2]])
            sentiment_test.append(dict_senti[element[3]])
            if element[5] is None:
                role_test.append(0)
            else:
                role_test.append(element[5])
            creator_test.append(element[4])
            freq_test.append(int(element[6]))
        # 读取训练集数据
        mycursor.execute(
            "SELECT preprocessed_summary, preprocessed_description, resolution, sentiment, creator, creator_developer, creator_freq from enhancement where sentiment is not null and product!='" +
            product_list[fold_id] + "'")
        myresult = mycursor.fetchall()
        result = myresult[:]

        for index, element in enumerate(result):
            summary_train.append(element[0].replace("\n", " "))
            descrip_train.append(element[1].replace("\n", " "))
            y_train.append(dict[element[2]])
            sentiment_train.append(dict_senti[element[3]])
            if element[5] is None:
                role_train.append(0)
            else:
                role_train.append(element[5])
            creator_train.append(element[4])
            freq_train.append(element[6])

    return summary_train, summary_test, descrip_train, descrip_test, y_train, y_test, sentiment_train, sentiment_test, creator_train, creator_test, role_train, role_test, freq_train, freq_test

sg = 1
window = 7
min_count = 0
negative = 5
sample = 0.00025
hs = 1

MAX_SEQUENCE_LENGTH_s = 30
MAX_SEQUENCE_LENGTH_d = 170
EMBEDDING_DIM = 100  # 词向量的大小
epoch = 20
batch_size = 640

# 对特征随机过采样或者欠采样
def over_under_sample(summary_train, description_train, sentiment_train, role_train, creator_train, freq_train, y_train, over = True):
    summary, descrip, sentiment, role, creator, freq = [], [], [], [], [], []
    length = len(summary_train)
    index = np.array(list(range(length))).reshape(length, 1)
    if over:
        ros = RandomOverSampler(random_state=0)
        index, y_train = ros.fit_resample(index, y_train)
    else:
        rus = RandomUnderSampler(random_state=0)
        index, y_train = rus.fit_resample(index, y_train)
    for i in index.reshape(length).tolist():
        summary.append(summary_train[i])
        descrip.append(description_train[i])
        sentiment.append(sentiment_train[i])
        role.append(role_train[i])
        creator.append(creator_train[i])
        freq.append(freq_train[i])
    summary_train, description_train, sentiment_train, role_train, creator_train, freq_train = np.array(summary), np.array(descrip), np.array(sentiment), np.array(role), np.array(creator), np.array(freq)
    return summary_train, description_train, sentiment_train, role_train, creator_train, freq_train, y_train

def CNN(summary_train, summary_test, descrip_train, descrip_test, y_train, y_test, sentiment_train, sentiment_test, creator_train, creator_test, role_train, role_test, freq_train, freq_test):
    K.clear_session()
    # 首先将author和sentiment向量化
    role = np_utils.to_categorical(role_train+role_test)
    role_train = role[:len(role_train)]
    role_test = role[len(role_train):]
    sentiment = np_utils.to_categorical(sentiment_train+sentiment_test)
    sentiment_train = sentiment[:len(sentiment_train)]
    sentiment_test = sentiment[len(sentiment_train):]
    sentiment_train = torch.tensor(sentiment_train)
    sentiment_test = torch.tensor(sentiment_test)
    sentiment_train = sentiment_train.unsqueeze(1)
    sentiment_test = sentiment_test.unsqueeze(1)

    encoder = LabelEncoder()
    encoder = encoder.fit_transform(creator_train+creator_test)
    creator = np_utils.to_categorical(encoder)
    creator_train = creator[:len(creator_train)]
    creator_test = creator[len(creator_train):]

    # 打乱顺序
    summary_train, summary_test, descrip_train, descrip_test, y_train, y_test = np.array(summary_train), np.array(summary_test), np.array(descrip_train), np.array(descrip_test), np.array(y_train), np.array(y_test)
    sentiment_train, sentiment_test, creator_train, creator_test, role_train, role_test, freq_train, freq_test = np.array(sentiment_train), np.array(sentiment_test), np.array(creator_train), np.array(creator_test), np.array(role_train), np.array(role_test), np.array(freq_train), np.array(freq_test)
    
    index = [i for i in range(len(summary_train))] 
    random.shuffle(index)
    summary_train = summary_train[index]
    descrip_train = descrip_train[index]
    y_train = y_train[index]
    sentiment_train = sentiment_train[index]
    creator_train = creator_train[index]
    role_train = role_train[index]
    freq_train = freq_train[index]
    # print(len(pos_train[0]))

    index = [i for i in range(len(summary_test))] 
    random.shuffle(index)
    summary_test = summary_test[index]
    summary_test = summary_test[index]
    y_test = y_test[index]
    sentiment_test = sentiment_test[index]
    creator_test = creator_test[index]
    role_test = role_test[index]
    creator_test = creator_test[index]
    freq_test = freq_test[index]

    text_train, text_test = [], []
    for index, element in enumerate(summary_train):
        text_train.append(summary_train[index] + " " + descrip_train[index])


    print("Tokenizer----------")
    tokenizer_s = Tokenizer()
    tokenizer_s.fit_on_texts(np.concatenate((summary_train, summary_test),axis=0))
    word_index_s = tokenizer_s.word_index
    print("Pad_sequence----------")
    summary_train = tokenizer_s.texts_to_sequences(summary_train)
    summary_train = pad_sequences(summary_train, maxlen=MAX_SEQUENCE_LENGTH_s)
    summary_test = tokenizer_s.texts_to_sequences(summary_test)
    summary_test = pad_sequences(summary_test, maxlen=MAX_SEQUENCE_LENGTH_s)

    tokenizer_d = Tokenizer()
    tokenizer_d.fit_on_texts(np.concatenate((descrip_train, descrip_test), axis=0))
    word_index_d = tokenizer_d.word_index
    description_train = tokenizer_d.texts_to_sequences(descrip_train)
    description_train = pad_sequences(description_train, maxlen=MAX_SEQUENCE_LENGTH_d)
    description_test = tokenizer_d.texts_to_sequences(descrip_test)
    description_test = pad_sequences(description_test, maxlen=MAX_SEQUENCE_LENGTH_d)

    print('Embedding----------')
    nb_words_s = len(word_index_s)
    embedding_matrix_s = np.zeros((nb_words_s + 1, EMBEDDING_DIM))

    nb_words_d = len(word_index_d)
    embedding_matrix_d = np.zeros((nb_words_d + 1, EMBEDDING_DIM))

    # 加载glove字典
    embeddings_index = {}
    f = open(os.path.join('/home/chenheng/ch/classifier/utils/glove.6B/', 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    for word, i in word_index_s.items():
        try:
            embedding_vector_s = embeddings_index.get(word)
        except KeyError:
            continue
        if embedding_vector_s is not None:
            embedding_matrix_s[i] = embedding_vector_s

    for word, i in word_index_d.items():
        try:
            embedding_vector_d = embeddings_index.get(word)
        except KeyError:
            continue
        if embedding_vector_d is not None:
            embedding_matrix_d[i] = embedding_vector_d

    # embedding层
    embedding_layer_s = Embedding(nb_words_s + 1, EMBEDDING_DIM, weights=[embedding_matrix_s],
                                  input_length=MAX_SEQUENCE_LENGTH_s, trainable=False)
    embedding_layer_d = Embedding(nb_words_d + 1, EMBEDDING_DIM, weights=[embedding_matrix_d],
                                  input_length=MAX_SEQUENCE_LENGTH_d, trainable=False)

    inputs_s = Input(shape=(MAX_SEQUENCE_LENGTH_s,), name='inputs_s')
    inputs_d = Input(shape=(MAX_SEQUENCE_LENGTH_d,), name='input_d')
    senti_input = Input(shape=(sentiment_train.shape[1], sentiment_train.shape[2],), name='senti_input')
    role_inputs = Input(shape=(3,), name='role_inputs')
    creator_inputs = Input(shape=(len(creator_train[0]),), name='creator_inputs')
    freq_inputs = Input(shape=(1,), name='freq_inputs')

    sentence_input_s = embedding_layer_s(inputs_s)
    sentence_input_d = embedding_layer_d(inputs_d)

    conv1D_s = (Conv1D(64, 1, padding='same', activation='relu'))(sentence_input_s)
    conv1D_d = (Conv1D(64, 1, padding='same', activation='relu'))(sentence_input_d)
    conv1D_senti = (Conv1D(64, 1, padding='same', activation='relu'))(senti_input)
    conv1D_s = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                  beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
                                  moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                  beta_constraint=None, gamma_constraint=None)(conv1D_s)
    conv1D_d = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                  beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
                                  moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                  beta_constraint=None, gamma_constraint=None)(conv1D_d)
    conv1D_senti = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                      beta_initializer='zeros', gamma_initializer='ones',
                                      moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                      beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                                      gamma_constraint=None)(conv1D_senti)

    conv1D_s = Flatten()(conv1D_s)
    conv1D_d = Flatten()(conv1D_d)
    conv1D_senti = Flatten()(conv1D_senti)
    # concatenation = concatenate([conv1D_s, conv1D_d, creator_inputs, role_inputs, freq_inputs])#加creator profile 3
    # concatenation = concatenate([conv1D_s, conv1D_d, conv1D_senti])#加情感分类 2
    concatenation = concatenate([conv1D_s, conv1D_d])#text 1

    attention_probs = Dense(12800, activation='softmax', name='attention_vec')(concatenation)
    attention_mul = Multiply()([concatenation, attention_probs])

    dropout = Dropout(0.5, noise_shape=None, seed=None)(attention_mul)
    dropout = Dense(512)(dropout)
    dropout = Dense(32)(dropout)
    predictions = Dense(1, activation='sigmoid')(dropout)

    adam = Adam(lr=0.0001)
    model = Model(inputs=[inputs_s, inputs_d, senti_input, role_inputs, creator_inputs, freq_inputs], outputs=predictions)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()
    early_stopping = EarlyStopping(monitor='acc', patience=3)
    reduceLROnPlateau = ReduceLROnPlateau(monitor='acc', factor=0.1, patience=3, mode='auto',
                                          min_delta=0.001, cooldown=0, min_lr=0)
    # history = model.fit([summary_train, description_train, sentiment_train], y_train, batch_size=batch_size, epochs=epoch, verbose=1, callbacks=[early_stopping, reduceLROnPlateau])
    history = model.fit([summary_train, description_train, sentiment_train, role_train, creator_train, freq_train], y_train, batch_size=batch_size,
                        epochs=epoch, verbose=1, callbacks=[early_stopping, reduceLROnPlateau], validation_data=([summary_test, description_test, sentiment_test, role_test, creator_test, freq_test], y_test))
    score = model.evaluate([summary_test, description_test, sentiment_test, role_test, creator_test, freq_test], y_test, verbose=1)
    y_predict = model.predict([summary_test, description_test, sentiment_test, role_test, creator_test, freq_test])

    for index, value in enumerate(y_predict):
        if (value >= 0.5):
            y_predict[index] = 1
        elif (value < 0.5):
            y_predict[index] = 0

    precision, recall, f, support = metrics.precision_recall_fscore_support(y_test, y_predict, beta=1.0, labels=None, pos_label=1, average=None, warn_for=('precision', 'recall', 'f-score'), sample_weight=None)
    
    cm = metrics.confusion_matrix(y_test, y_predict)
    print(cm)

    return precision, recall, f, score


total_acc, total_p0, total_r0, total_f0, total_p1, total_r1, total_f1 = 0, 0, 0, 0, 0, 0, 0
FLAG = 0  # 设置
for fold_id in range(10):
    summary_train, summary_test, descrip_train, descrip_test, y_train, y_test, sentiment_train, sentiment_test, creator_train, creator_test, role_train, role_test, freq_train, freq_test = read_data(fold_id, FLAG)
    # print(x_train.shape)

    precision, recall, f, score = CNN(summary_train, summary_test, descrip_train, descrip_test, y_train, y_test, sentiment_train, sentiment_test, creator_train, creator_test, role_train, role_test, freq_train, freq_test)

    total_acc = total_acc + score[1]
    total_p1 = total_p1 + precision[1]
    total_r1 = total_r1 + recall[1]
    total_f1 = total_f1 + f[1]
    total_p0 = total_p0 + precision[0]
    total_r0 = total_r0 + recall[0]
    total_f0 = total_f0 + f[0]
    print(str(precision[1])+" "+str(recall[1]) + " " + str(f[1]) + " " +str(score))
print(str(float(total_acc / 10)) + " " + str(float(total_p0 / 10)) + " " + str(float(total_r0 / 10)) + " " + str(
    float(total_f0 / 10)) + " " + str(float(total_p1 / 10)) + " " + str(float(total_r1 / 10)) + " " + str(
    float(total_f1 / 10)))
# 十折交叉验证，glove
import numpy as np
import warnings
import os
from gensim.models import Word2Vec
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import sequence
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from tensorflow.keras import Model
from keras.optimizers import Adam
import tensorflow as tf

from tensorflow.keras.layers import Input, Embedding, Dense, Embedding, Dense, CuDNNLSTM, Bidirectional, Dropout, Concatenate
warnings.filterwarnings("ignore")

max_features = 5000
maxlen = 200
batch_size = 640
embedding_dims = 100
epochs = 20
class_num=1

embed = "glove"
def LSTM_model(x_train, x_test, y_train, y_test, sentiment_train, sentiment_test, author_train, author_test, creator_train, creator_test, freq_train, freq_test):

    # W2V_MODEL = Word2Vec(x_train, sg=1, vector_size=embedding_dims, window=7, min_count=0, negative=5, sample=0.00025, hs=1)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(np.concatenate((x_train,x_test),axis=0))
    word_index = tokenizer.word_index
    x_train = tokenizer.texts_to_sequences(x_train)
    x_train = pad_sequences(x_train, maxlen=maxlen)
    x_test = tokenizer.texts_to_sequences(x_test)
    x_test = pad_sequences(x_test, maxlen=maxlen)

    if embed == "word2vec":
            # embedding
        nb_words = len(word_index)
        embedding_matrix = np.zeros((nb_words + 1, embedding_dims))

        for word, i in word_index.items():
            try:
                embedding_vector = W2V_MODEL.wv[word]
            except KeyError:
                continue
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    elif embed =="glove":
        # 加载glove字典
        embeddings_index = {}
        f = open(os.path.join('/home/chenheng/ch/classifier/utils/glove.6B/', 'glove.6B.100d.txt'))
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        
        nb_words = len(word_index)
        embedding_matrix = np.zeros((nb_words + 1, embedding_dims))

        for word, i in word_index.items():
            try:
                embedding_vector = embeddings_index.get(word)
            except KeyError:
                continue
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector


    # embedding层
    embedding_layer = Embedding(nb_words + 1,embedding_dims,weights=[embedding_matrix],input_length=maxlen,trainable=False)

    # embedding_layer = Embedding(max_features, embedding_dims, input_length=maxlen)

    inputs = Input(shape=(maxlen,), name='inputs')
    author_inputs = Input(shape=(3,), name='author_inputs')
    sentiment_inputs = Input(shape=(3,), name='sentiment_inputs')
    creator_inputs = Input(shape=(len(creator_train[0]),), name='creator_inputs')
    freq_inputs = Input(shape=(1,), name='freq_inputs')

    embedding = embedding_layer(inputs)
    x = Bidirectional(CuDNNLSTM(128))(embedding)
    x = Dropout(0.5)(x)
    # x = Concatenate()([x,sentiment_inputs])
    x = Concatenate()([x, creator_inputs, author_inputs, freq_inputs])
    x = Dense(512)(x)
    x = Dense(32)(x)

    output = Dense(class_num, activation='sigmoid')(x)

    # adam = Adam(lr=0.0001)
    adam = tf.keras.optimizers.Adam(lr = 0.0001)
    model = Model(inputs=[inputs, author_inputs, sentiment_inputs, creator_inputs, freq_inputs], outputs=output)
    model.compile(adam, 'binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # print('Train...')
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max')
    model.fit([x_train,author_train, sentiment_train, creator_train, freq_train], y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # callbacks=[early_stopping],
            # class_weight = class_weights,
            validation_data=([x_test, author_test, sentiment_test, creator_test, freq_test], y_test))

    # print('Test...')
    score = model.evaluate([x_test, author_test, sentiment_test, creator_test, freq_test], y_test, verbose=1)
    y_predict = model.predict([x_test, author_test, sentiment_test, creator_test, freq_test])

    return y_test, y_predict, score

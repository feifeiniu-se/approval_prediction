# 本文件粘贴自222服务器/home/chenheng/ch/classifier/DL_classifier/DBMethod/TextCNN_10fold_glove_copy.py
# 十折交叉验证，glove
import numpy as np
import random
import pymysql
import warnings
import os
import time
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import sequence
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.keras import Model
from collections import Counter
from tqdm import tqdm
from sklearn import model_selection, metrics
from tensorflow.keras.layers import Input, Embedding, Dense, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout, Multiply
from keras.utils import np_utils
from sklearn.utils import class_weight, resample
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from gensim.models import Word2Vec

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

warnings.filterwarnings("ignore")


max_features = 5000
maxlen = 200
batch_size = 32
embedding_dims = 100
epochs = 14
kernel_sizes=[3, 4, 5]
class_num=1
attention_flag = False
resample_flag = False # 是否进行重采样
over = False # 是否过采样

enhancement_len = 0

def pos_transfer(stanza_pos):
    x = []
    for i in stanza_pos:
        x.append(i.split(" "))
    # print(x[:10])
    new_x = []
    for i in x:
        # print(i)
        i = [int(var) for var in i]
        adj = i[9]+i[10]+i[11]
        modal = i[13]
        noun = i[15]+i[16]+i[17]+i[18]
        verb = i[30]+i[31]+i[32]+i[33]+i[34]+i[35]
        adv = i[23]+i[24]+i[25]+i[26]
        wh = i[36]+i[37]+i[38]+i[39]
        new_x.append([adj, modal, noun, verb, adv, wh])
    # print(new_x[:10])
    return new_x

# 对特征随机过采样或者欠采样
def over_under_sample(x_train,author_train, sentiment_train, creator_train, freq_train, y_train, over = True):
    
    x, author, sentiment, pos, creator, freq = [], [], [], [], [], []
    length = x_train.shape[0]
    index = np.array(list(range(length))).reshape(length, 1)
    counter = Counter(y_train.tolist())
    print('采样前样本比例： ',counter)
    if over:
        ros = RandomOverSampler(random_state=0)
        index, y_train = ros.fit_resample(index, y_train)
    else:
        rus = RandomUnderSampler(random_state=0)
        index, y_train = rus.fit_resample(index, y_train)
    print('=====================')
    counter = Counter(y_train.tolist())
    print('采样后样本比例： ', counter)
    for i in index.reshape(index.shape[0]).tolist():
        x.append(x_train[i])
        author.append(author_train[i])
        sentiment.append(sentiment_train[i])
        creator.append(creator_train[i])
        freq.append(freq_train[i])
    x_train,author_train, sentiment_train, creator_train, freq_train = np.array(x), np.array(author), np.array(sentiment), np.array(creator), np.array(freq)
    return x_train,author_train, sentiment_train, creator_train, freq_train, y_train

def CNN_model(x_train, x_test, y_train, y_test, sentiment_train, sentiment_test, author_train, author_test, creator_train, creator_test, freq_train, freq_test):

    # 首先将author和sentiment向量化
    author = np_utils.to_categorical(author_train+author_test)
    author_train = author[:len(author_train)]
    author_test = author[len(author_train):]
    sentiment = np_utils.to_categorical(sentiment_train+sentiment_test)
    sentiment_train = sentiment[:len(sentiment_train)]
    sentiment_test = sentiment[len(sentiment_train):]

    encoder = LabelEncoder()
    encoder = encoder.fit_transform(creator_train+creator_test)
    # 更换creator信息的嵌入方式
    # creator = np_utils.to_categorical(encoder) # 使用defect表中creator信息会导致该部分参数量非常稀疏(348440个开发者名字)，所以改为稠密向量嵌入
    creator = encoder # 
    creator_train = creator[:len(creator_train)]
    creator_test = creator[len(creator_train):]

    x_train, x_test, y_train, y_test = np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)
    author_train, author_test, creator_train, creator_test, freq_train, freq_test = np.array(author_train), np.array(author_test), np.array(creator_train), np.array(creator_test), np.array(freq_train), np.array(freq_test)
    sentiment_train, sentiment_test = np.array(sentiment_train), np.array(sentiment_test)
    #打乱顺序
    index = [i for i in range(len(x_train))]
    random.shuffle(index)
    x_train = x_train[index]
    y_train = y_train[index]
    sentiment_train = sentiment_train[index]
    author_train = author_train[index]
    # pos_train = pos_train[index]
    creator_train = creator_train[index]
    freq_train = freq_train[index]
    # print(len(pos_train[0]))

    index = [i for i in range(len(x_test))] 
    random.shuffle(index)
    x_test = x_test[index]
    y_test = y_test[index]
    sentiment_test = sentiment_test[index]
    author_test = author_test[index]
    # pos_test = pos_test[index]
    creator_test = creator_test[index]
    freq_test = freq_test[index]

    W2V_MODEL = Word2Vec(x_train, sg=0, vector_size=embedding_dims, window=7, min_count=0, negative=5, sample=0.00025, hs=1)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(np.concatenate((x_train,x_test),axis=0))
    word_index = tokenizer.word_index
    x_train = tokenizer.texts_to_sequences(x_train)
    x_train = pad_sequences(x_train, maxlen=maxlen)
    x_test = tokenizer.texts_to_sequences(x_test)
    x_test = pad_sequences(x_test, maxlen=maxlen)
    # print('x_train shape:', x_train.shape)
    # print('x_test shape:', x_test.shape)

    # y_train = np.array(y_train)
    # y_test = np.array(y_test)

    # class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

    # embedding
    nb_words = len(word_index)
    embedding_matrix = np.zeros((nb_words + 1, embedding_dims))

    # 加载glove字典
    embeddings_index = {}
    f = open(os.path.join('/home/niufeifei/glove/', 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))

    for word, i in word_index.items():
        try:
            embedding_vector = W2V_MODEL.wv[word]
            # embedding_vector = embeddings_index.get(word)
        except KeyError:
            continue
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # embedding层
    embedding_layer = Embedding(nb_words + 1,embedding_dims,weights=[embedding_matrix],input_length=maxlen,trainable=False)

    # 更换creator信息的嵌入方式
    creator_embedding_layer = Embedding(348440, 300, trainable=True) #

    inputs = Input(shape=(maxlen,), name='inputs')
    author_inputs = Input(shape=(3,), name='author_inputs')
    sentiment_inputs = Input(shape=(3,), name='sentiment_inputs')
    # pos_inputs = Input(shape=(6,), name='pos_inputs')

    # 更换creator信息的嵌入方式
    # creator_inputs = Input(shape=(len(creator_train[0]),), name='creator_inputs')
    creator_inputs = Input(shape=(1,), name='creator_inputs')  # 
    creator_inputs_embed = tf.squeeze(creator_embedding_layer(creator_inputs), [1]) # 
    print("here:")
    print(creator_inputs_embed.shape) #

    freq_inputs = Input(shape=(1,), name='freq_inputs')
    embedding = embedding_layer(inputs)
    
    convs = []
    for kernel_size in kernel_sizes:
        convs_1 = Conv1D(1024, kernel_size, activation='relu')(embedding)
        max_pooling_1 = GlobalMaxPooling1D()(convs_1)
        convs.append(max_pooling_1)
    x = Concatenate()(convs)
    x = Dropout(0.2)(x)

    # 更换creator信息的嵌入方式
    # x = Concatenate()([x, author_inputs])
    # x = Concatenate()([x, creator_inputs, author_inputs, freq_inputs])
    x = Concatenate()([x, creator_inputs_embed, author_inputs, freq_inputs]) # 

    # x = Concatenate()([x,sentiment_inputs])
    # x = Dense(32)(x)
    # print(x.shape)

    # 增加dense注意力
    if attention_flag:
        attention_probs = Dense(387, activation='softmax', name='attention_vec')(x)
        attention_mul =  Multiply()([x, attention_probs])
        output = Dense(class_num, activation='sigmoid')(attention_mul)
    else:
        output = Dense(class_num, activation='sigmoid')(x)

    # adam = tf.keras.optimizers.adam(lr = 0.0001)
    model = Model(inputs=[inputs, author_inputs, sentiment_inputs, creator_inputs, freq_inputs], outputs=output)
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    model.summary()
    # quit()

    # 过采样或者欠采样
    if resample_flag:
        x_train,author_train, sentiment_train, creator_train, freq_train, y_train = over_under_sample(x_train,author_train, sentiment_train, creator_train, freq_train, y_train, over = over)
    
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

    for index, value in enumerate(y_predict):
        if(value>=0.5):
            y_predict[index] = 1
        elif(value<0.5):
            y_predict[index] = 0

    # print(metrics.precision_recall_fscore_support(y_test, y_predict, beta=1.0, labels=None, pos_label=1, average=None, warn_for=('precision', 'recall', 'f-score'), sample_weight=None))
    # print("score: ", score)
    precision, recall, f, support = metrics.precision_recall_fscore_support(y_test, y_predict, beta=1.0, labels=None, pos_label=1, average=None, warn_for=('precision', 'recall', 'f-score'), sample_weight=None)
    
    cm = metrics.confusion_matrix(y_test, y_predict)
    print(cm)

    return precision, recall, f, score

def read_data(fold_id, FLAG):
    # print("Loading data...")
    
    x_train, x_test, y_train, y_test, sentiment_train, sentiment_test, author_train, author_test, creator_train, creator_test, freq_train, freq_test = [], [], [], [], [], [], [], [], [], [], [], []
    dict = {"FIXED": 1, "INVALID": 0, "DUPLICATE": 0, "WONTFIX": 0, "INCOMPLETE": 0, "WORKSFORME": 0, "EXPIRED": 0, "MOVED": 0, "INACTIVE": 0}
    dict_senti = {"negative": 0, "positive": 1, "neutral": 2}

    db = pymysql.connect(
        host="172.29.7.222",
        port=3306,
        user="root",
        password="1234",
        database="bugzilla"
    )
    mycursor = db.cursor()

    # 读取defect表中的数据作为训练集的一部分
    mycursor.execute("SELECT stanza, resolution, sentiment, creator_developer_two, creator, creator_freq_two from defect where resolution is not null and stanza is not null and creation_time between '1997-09-10' and '2016-07-13'")
    mydefect_train = mycursor.fetchall()
    defect_train = mydefect_train[:]

    if FLAG==0:#随机平均划分十折
        # 读取测试集数据
        mycursor.execute("SELECT stanza, resolution, sentiment, creator_developer_two, creator, creator_freq_two from enhancement where folds_id=" + str(fold_id))
        myresult = mycursor.fetchall()
        result = myresult[:]
        
        for index, element in enumerate(result):
            # x_test.append(element[0] + " " + " ".join(eval(element[5])[fold_id]))
            x_test.append(" ".join(eval(element[0]))) # stanza
            y_test.append(dict[element[1]])
            sentiment_test.append(dict_senti[element[2]])
            if element[3] is None:
                author_test.append(0)
            else:
                author_test.append(element[3])
            creator_test.append(element[4])
            freq_test.append(int(element[5]))
        # 读取训练集数据
        mycursor.execute("SELECT stanza, resolution, sentiment, creator_developer_two, creator, creator_freq_two from enhancement where folds_id is not null and folds_id!=" + str(fold_id))
        myresult = mycursor.fetchall()
        # result = myresult[:]
        enhancement_len = len(result)
        result = myresult[:] + defect_train[:]

        for index, element in enumerate(result):
            x_train.append(" ".join(eval(element[0])))
            y_train.append(dict[element[1]])
            sentiment_train.append(dict_senti[element[2]])
            if element[3] is None:
                author_train.append(0)
            else:
                author_train.append(element[3])
            creator_train.append(element[4])
            freq_train.append(int(element[5]))

    if FLAG==1:#十个项目划分十折
        product_list = ["Bugzilla", "SeaMonkey", "Core Graveyard", "Core", "MailNews Core", "Toolkit", "Firefox", "Thunderbird", "Calendar", "Camino Graveyard"]
        # 读取测试集数据
        mycursor.execute("SELECT stanza, resolution, sentiment, creator_developer_two, creator, creator_freq_two from enhancement where sentiment is not null and product='" + product_list[fold_id] + "'")
        myresult = mycursor.fetchall()
        result = myresult[:]
        
        for index, element in enumerate(result):
            x_test.append(element[0])
            y_test.append(dict[element[1]])
            sentiment_test.append(dict_senti[element[2]])
            if element[3] is None:
                author_test.append(0)
            else:
                author_test.append(element[3])
            creator_test.append(element[4])
            freq_test.append(int(element[5]))
        # 读取训练集数据
        mycursor.execute("SELECT stanza, resolution, sentiment, creator_developer_two, creator, creator_freq_two from enhancement where sentiment is not null and product!='" + product_list[fold_id] + "'")
        myresult = mycursor.fetchall()
        result = myresult[:] + defect_train[:]

        for index, element in enumerate(result):
            x_train.append(element[0])
            y_train.append(dict[element[1]])
            sentiment_train.append(dict_senti[element[2]])
            if element[3] is None:
                author_train.append(0)
            else:
                author_train.append(element[3])
            creator_train.append(element[4])
            freq_train.append(int(element[5]))
        # print("X_train length: " + str(len(x_train)))
    # print("X_test length: " + str(len(x_test)))

    print(len(x_train))
    print(len(x_test))
    return x_train, x_test, y_train, y_test, sentiment_train, sentiment_test, author_train, author_test, creator_train, creator_test, freq_train, freq_test

total_acc, total_p1, total_r1, total_f1, total_p0, total_r0, total_f0 = 0, 0, 0, 0, 0, 0, 0
FLAG = 0
time_start = time.time() #开始计时
valid_file = open('./valid.txt', 'a')
for fold_id in range(0,7):
    x_train, x_test, y_train, y_test, sentiment_train, sentiment_test, author_train, author_test, creator_train, creator_test, freq_train, freq_test = read_data(fold_id, FLAG)
    # print(x_test[:10])
    precision, recall, f, score = CNN_model(x_train, x_test, y_train, y_test, sentiment_train, sentiment_test, author_train, author_test, creator_train, creator_test, freq_train, freq_test)
    total_acc = total_acc + score[1]
    total_p1 = total_p1 + precision[1]
    total_p0 = total_p0 + precision[0]
    total_r1 = total_r1 + recall[1]
    total_r0 = total_r0 + recall[0]
    total_f1 = total_f1 + f[1]
    total_f0 = total_f0 + f[0]

    print("flag: " + str(FLAG))
    valid_file.write('fold_id为：' + str(fold_id) + "\t" + str(precision[1])+"\t"+str(recall[1]) + "\t" + str(f[1]) + "\t" + str(precision[0])+"\t"+str(recall[0]) + "\t" + str(f[0]) + "\t" +str(score[1]) + "\n")
    print(str(precision[1])+" "+str(recall[1]) + " " + str(f[1]) + " " + str(precision[0])+" "+str(recall[0]) + " " + str(f[0]) + " " +str(score[1]))
time_end = time.time()    #结束计时
time_c= time_end - time_start   #运行所花时间
print('time cost', time_c, 's')
print(str(float(total_acc/10)) + " " + str(float(total_p1/10)) + " " + str(float(total_r1/10)) + " " + str(float(total_f1/10)) + " " + str(float(total_p0/10)) + " " + str(float(total_r0/10)) + " " + str(float(total_f0/10)))
valid_file.write(str(float(total_acc/10)) + " " + str(float(total_p1/10)) + " " + str(float(total_r1/10)) + " " + str(float(total_f1/10)) + " " + str(float(total_p0/10)) + " " + str(float(total_r0/10)) + " " + str(float(total_f0/10)) + "\n")

valid_file.close()

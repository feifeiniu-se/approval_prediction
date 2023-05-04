from enum import Flag
import numpy
import pymysql
# import sklearn
# from tqdm import tqdm
# import os
# import pandas as pd
#
# from baseline.Nizamani import MNB_model
# from baseline.Umer import SVM_model
from utils import read_data, VSM, train
from sklearn.preprocessing import LabelEncoder
import numpy as np
from keras.utils import np_utils

def normalization(train, test):
    '''
     归一化数据
     input: (numpy.array, numpy.array)
     return: (numpy.array, numpy.array)
    '''
    Min = np.min(train)
    Max = np.max(train)
    train = (train - Min) / (Max - Min)
    test = (test - Min) / (Max - Min)
    return train, test


def read_data(fold_id, FLAG):
    # print("Loading data...")

    x_train, x_test, y_train, y_test, sentiment_train, sentiment_test, author_train, author_test, pos_train, pos_test, creator_train, creator_test, freq_train, freq_test, bert_train, bert_test = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
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
    # 随机平均划分十折
    if FLAG == 0: 
        # 读取测试集数据
        mycursor.execute(
            "SELECT preprocessed12345, stanza_pos, resolution, sentiment, creator_developer, creator, creator_freq, bert_vector from enhancement where folds_id=" + str(
                fold_id))
        myresult = mycursor.fetchall()
        result = myresult[:]

        for index, element in enumerate(result):
            # x_test.append(" ".join(list(eval(element[0]).keys())))
            x_test.append(element[0])
            pos_test.append(element[1])
            # x_test.append(element[0].replace("\n", " ") + " " + element[1].replace("\n", " ") + " " + element[5].replace("\n", " "))
            y_test.append(dict[element[2]])
            sentiment_test.append(dict_senti[element[3]])
            if element[4] is None:
                author_test.append(0)
            else:
                author_test.append(element[4])
            creator_test.append(element[5])
            freq_test.append(int(element[6]))
            tmp = element[7].replace("\n", " ").replace("[", "").replace("]", "").split()
            tmp = [float(x) for x in tmp]
            bert_test.append(tmp)
        # 读取训练集数据
        mycursor.execute(
            "SELECT preprocessed12345, stanza_pos, resolution, sentiment, creator_developer, creator, creator_freq, bert_vector from enhancement where folds_id is not null and folds_id!=" + str(
                fold_id))
        myresult = mycursor.fetchall()
        result = myresult[:]

        for index, element in enumerate(result):
            x_train.append(element[0])
            pos_train.append(element[1])
            y_train.append(dict[element[2]])
            sentiment_train.append(dict_senti[element[3]])
            if element[4] is None:
                author_train.append(0)
            else:
                author_train.append(element[4])
            creator_train.append(element[5])
            freq_train.append(element[6])
            tmp = element[7].replace("\n", " ").replace("[", "").replace("]", "").split()
            tmp = [float(x) for x in tmp]
            bert_train.append(tmp)
    if FLAG == 1:  # 十个项目划分十折
        product_list = ["Bugzilla", "SeaMonkey", "Core Graveyard", "Core", "MailNews Core", "Toolkit", "Firefox",
                        "Thunderbird", "Calendar", "Camino Graveyard"]
        # 读取测试集数据
        mycursor.execute(
            "SELECT preprocessed12345, stanza_pos, resolution, sentiment, creator_developer, creator, creator_freq, bert_vector from enhancement where sentiment is not null and product='" +
            product_list[fold_id] + "'")
        myresult = mycursor.fetchall()
        result = myresult[:]

        for index, element in enumerate(result):
            x_test.append(element[0])
            pos_test.append(element[1])
            y_test.append(dict[element[2]])
            sentiment_test.append(dict_senti[element[3]])
            if element[4] is None:
                author_test.append(0)
            else:
                author_test.append(element[4])
            creator_test.append(element[5])
            freq_test.append(int(element[6]))
            tmp = element[7].replace("\n", " ").replace("[", "").replace("]", "").split()
            tmp = [float(x) for x in tmp]
            bert_test.append(tmp)
        # 读取训练集数据
        mycursor.execute(
            "SELECT preprocessed12345, stanza_pos, resolution, sentiment, creator_developer, creator, creator_freq, bert_vector from enhancement where sentiment is not null and product!='" +
            product_list[fold_id] + "'")
        myresult = mycursor.fetchall()
        result = myresult[:]

        for index, element in enumerate(result):
            x_train.append(element[0])
            pos_train.append(element[1])
            y_train.append(dict[element[2]])
            sentiment_train.append(dict_senti[element[3]])
            if element[4] is None:
                author_train.append(0)
            else:
                author_train.append(element[4])
            creator_train.append(element[5])
            freq_train.append(element[6])
            tmp = element[7].replace("\n", " ").replace("[", "").replace("]", "").split()
            tmp = [float(x) for x in tmp]
            bert_train.append(tmp)
        # print("X_train length: " + str(len(x_train)))
    # print("X_test length: " + str(len(x_test)))

    return x_train, x_test, y_train, y_test, sentiment_train, sentiment_test, author_train, author_test, pos_train, pos_test, creator_train, creator_test, freq_train, freq_test, bert_train, bert_test

def run(FLAG, FEATURE, MODEL, SENTIMENT, CREATOR, fin):
    total_acc, total_p0, total_r0, total_f0, total_p1, total_r1, total_f1 = 0, 0, 0, 0, 0, 0, 0
    for fold_id in range(0,10):
        x_train, x_test, y_train, y_test, sentiment_train, sentiment_test, author_train, author_test, pos_train, pos_test, creator_train, creator_test, freq_train, freq_test, bert_train, bert_test = read_data(fold_id, FLAG)

        if FEATURE in ["TFIDF", "TF"]:
            x_train, x_test = VSM(FEATURE, x_train, x_test)#文本向量化
        elif FEATURE == "BERT":
            x_train, x_test = np.array(bert_train), np.array(bert_test)

        # print(x_train.shape)
        if SENTIMENT is True:
            sentiment = np_utils.to_categorical(sentiment_train+sentiment_test)
            sentiment_train = sentiment[:len(sentiment_train)]
            sentiment_test = sentiment[len(sentiment_train):]
            # print(sentiment_test[0])
            x_train = np.concatenate((sentiment_train, x_train), axis=1)
            x_test = np.concatenate((sentiment_test, x_test), axis=1)
    
        # print(x_train.shape)
        if CREATOR is True:
            # print("OK")
            #作者名字编码
            encoder = LabelEncoder()
            encoder = encoder.fit_transform(creator_train+creator_test)
            # creator = np_utils.to_categorical(encoder)
            creator = encoder # svm
            creator_train = creator[:len(creator_train)]
            creator_test = creator[len(creator_train):]
            #作者身份编码
            author = np_utils.to_categorical(author_train+author_test)
            author_train = author[:len(author_train)]
            author_test = author[len(author_train):]

            author_train, author_test, creator_train, creator_test, freq_train, freq_test = np.array(author_train), np.array(author_test), np.array(creator_train), np.array(creator_test), np.array(freq_train), np.array(freq_test)

            # 增加归一化
            # (creator_train, creator_test), (freq_train, freq_test) = normalization(creator_train, creator_test), normalization(freq_train, freq_test)

            x_train = np.concatenate((x_train, author_train, ), axis=1)
            x_test = np.concatenate((x_test, author_test, ), axis=1)
            x_train = np.column_stack((x_train, freq_train, creator_train))
            x_test = np.column_stack((x_test, freq_test, creator_test))

            (x_train, x_test) = normalization(x_train, x_test) # svm

        precision, recall, f, score = train(MODEL, x_train, x_test, y_train, y_test, confusion_matrix_file)
        total_acc = total_acc + score
        total_p1 = total_p1 + precision[1]
        total_r1 = total_r1 + recall[1]
        total_f1 = total_f1 + f[1]
        total_p0 = total_p0 + precision[0]
        total_r0 = total_r0 + recall[0]
        total_f0 = total_f0 + f[0]
        # print(str(precision[1])+" "+str(recall[1]) + " " + str(f[1]) + " " +str(score))
    print(str(float(total_acc / 10)) + " " + str(float(total_p1 / 10)) + " " + str(float(total_r1 / 10)) + " " + str(
        float(total_f1 / 10)) + " " + str(float(total_p0 / 10)) + " " + str(float(total_r0 / 10)) + " " + str(
        float(total_f0 / 10)))
    fin.write(str(float(total_acc / 10)) + " " + str(float(total_p1 / 10)) + " " + str(float(total_r1 / 10)) + " " + str(
        float(total_f1 / 10)) + " " + str(float(total_p0 / 10)) + " " + str(float(total_r0 / 10)) + " " + str(
        float(total_f0 / 10)) + "\n")
flags = [1]
# models = ["NN", "SVM", "MNB"] # MNB不可处理负特征，bert向量中存在负数换成高斯朴素贝叶斯
models = ["SVM"] # MNB不可处理负特征，bert向量中存在负数换成高斯朴素贝叶斯
sentiments = [False] # 不变
features = ["TF", "BERT"]
creators = [True]
# use_bert = [True, False] # 合并到features中
# 记录实验结果
res_path = './res.txt'
fin = open(res_path, 'w', encoding='utf-8')
confusion_matrix_file = open('./confusion_matrix.txt', 'w', encoding='utf-8')
for MODEL in models:
    for FLAG in flags:
        for SENTIMENT in sentiments:
            for FEATURE in features:
                for CREATOR in creators:
                    # for BERT in use_bert:
                    print("flag: "+str(FLAG) + " models: " + str(MODEL) + " sentiment: " + str(SENTIMENT) + " feature: " + str(FEATURE) + " creator: " + str(CREATOR))
                    fin.write("flag: "+str(FLAG) + " models: " + str(MODEL) + " sentiment: " + str(SENTIMENT) + " feature: " + str(FEATURE) + " creator: " + str(CREATOR) + '\n')
                    confusion_matrix_file.write("flag: "+str(FLAG) + " models: " + str(MODEL) + " sentiment: " + str(SENTIMENT) + " feature: " + str(FEATURE) + " creator: " + str(CREATOR) + '\n')
                    run(FLAG, FEATURE, MODEL, SENTIMENT, CREATOR, fin)

fin.close()
confusion_matrix_file.close()

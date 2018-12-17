#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import nltk
import numpy as np
import tensorflow as tf
import random
import pickle
import json
import constantes
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from keras.models import load_model
from ann_visualizer.visualize import ann_viz

class TensorFlowClassifier(object):
    model = None
    raw = None
    texts = list()
    labels = list()
    non_stop_words = set()
    sentences = list()
    stop_words = list()
    word2int = dict()
    int2word = dict()

    def __init__(self, dataset_path='conversas.txt'):

        with open(dataset_path, 'r') as arq:
            self.raw = arq.read().lower()

        lines = self.raw.split('\n')
        for line in lines:
            try:
                splited_line = line.split(',')
                sentence = ','.join(splited_line[:-2])
                self.labels.append(int(splited_line[-2]))
                self.sentences.append(sentence)
            except:
                pass

        stop_words = self.load_stop_words()
        sentences_nsw = [
            self.remove_stop_words(sentence) for sentence in self.sentences
        ]
        self.postagger = pickle.load(open('postagger.pickle', 'rb'))

        self.non_stop_words = set(' '.join(sentences_nsw).split())
        for i, word in enumerate(self.non_stop_words):
            self.word2int[word] = i
            self.int2word[i] = word

        for sent_ns in sentences_nsw:
            self.texts.append([self.word2int[word] for word in sent_ns.split()])
        
        self.load_model()
    
    def get_features(self, post, postagger):
        features = {}
        classes = []
        words = nltk.word_tokenize(post)
        q_words_count = 0
        err_words_count = 0
        ag_words_count = 0
        saud_words_count = 0
        neg_words_count = 0
        conf_words_count = 0
        q_words = ['que', 'quem', 'pode', 'posso', 'onde', 'como', 'q', 'porque', 'pq', 'qual']
        err_words = ['problema', 'erro', 'travando', 'funciona', 'funcionando', 'abre', 'trava', 'abrindo', 'abriu', 'abrir']
        ag_words = ['obrigada', 'obrigado', 'obg']
        saud_words = ['dia', 'tarde', 'noite', 'bom', 'boa', 'oi', 'ola', 'olá', 'oii']
        neg_words = ['não', 'nao', 'n']
        conf_words = ['é', 'sim', 'certo', 'pronto', 'ok', 'bem', 'beleza', 'blz', 'sei', 'entendo', 'entendido', 'entendi', 'já', 'ja']
        for word in words:
            classe = postagger.classify({'text': word})
            classes.append(classe)
            if word in q_words:
                q_words_count = q_words_count + 1
            if word in err_words:
                err_words_count = err_words_count + 1
            if word in ag_words:
                ag_words_count = ag_words_count + 1
            if word in saud_words:
                saud_words_count = saud_words_count + 1
            if word in neg_words:
                neg_words_count = neg_words_count + 1
            if word in conf_words:
                conf_words_count = conf_words_count + 1

        # features['pattern'] = ' '.join(classes)
        features['num_words'] = len(words)
        features['question_sign'] = 1 if '?' in post else 0
        features['err_words'] = err_words_count
        features['q_words'] = q_words_count
        features['ag_words'] = ag_words_count
        features['saud_words'] = saud_words_count
        features['neg_words'] = neg_words_count
        features['conf_words'] = conf_words_count

        if len(words) > 0:
            features['avg_word_size'] = len(post) / len(words)
        else:
            features['avg_word_size'] = 0
        
        return features
    
    def dict_to_array(self, features):
        return [features[key] for key in features.keys()]

    def get_data(self):
        result = list()
        for sent in self.sentences:
            features = self.get_features(sent, self.postagger)
            feature_values = self.dict_to_array(features)
            # feature_values.pop(0)
            result.append(feature_values)
        return result, self.labels

    def load_stop_words(self):
        return ['.', '?', '!', ':']

    def remove_stop_words(self, sentence):
        sent = sentence.replace(',', '')
        for sw in self.stop_words:
            sent = sent.replace(sw, '')
        return sent
    
    def load_model(self):
        
        texts, labels = self.get_data()
        texts, labels = shuffle(texts, labels)

        limit = 300
        (test_data, test_labels) = (texts[:limit], labels[:limit])
        (train_data, train_labels) = (texts[limit:], labels[limit:])

        train_labels = keras.utils.to_categorical(train_labels, num_classes=8)
        test_labels = keras.utils.to_categorical(test_labels, num_classes=8)

        train_data = np.asarray(train_data)
        test_data = np.asarray(test_data)

        model = keras.Sequential()
        model.add(keras.layers.Dense(32, activation='relu', input_dim=9))
        model.add(keras.layers.Dense(32, activation=tf.nn.relu))
        model.add(keras.layers.Dense(8, activation='softmax'))

        model.compile(
            optimizer=tf.train.AdamOptimizer(),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        history = model.fit(
            train_data,
            train_labels,
            epochs=80,
            verbose=0
        )
        self.model = model

        a, b = model.evaluate(test_data, test_labels)

        # print(a)
        os.system('clear')
        print("Classificador treinado em 80 épocas\nAcurácia: {}".format(b))
        previsao = model.predict(test_data)
        teste_matrix = [np.argmax(t) for t in test_labels]
        previsoes_matrix = [np.argmax(t) for t in previsao]

        # confusao = confusion_matrix(teste_matrix, previsoes_matrix)
        # print(confusao)
    
    def predict(self, text, format_result=False):
        features = self.get_features(text, self.postagger)
        features = self.dict_to_array(features)
        array = np.asarray([features])
        result = self.model.predict(array)[0]
        max_result = max(result)
        for i in range(len(result)):
            if result[i] == max_result:
                if format_result:
                    return constantes.label_to_text[i]
                return i

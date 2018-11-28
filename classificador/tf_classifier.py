import nltk
import numpy as np
import tensorflow as tf
import pickle
from tensorflow import keras
from random import shuffle
from sklearn.metrics import confusion_matrix

class TensorFlowClassifier(object):
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

        self.non_stop_words = set(' '.join(sentences_nsw).split())
        for i, word in enumerate(self.non_stop_words):
            self.word2int[word] = i
            self.int2word[i] = word

        for sent_ns in sentences_nsw:
            self.texts.append([self.word2int[word] for word in sent_ns.split()])
    
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

        features['pattern'] = ' '.join(classes)
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

    def get_data(self):
        # return self.texts, self.labels
        postagger = pickle.load(open('postagger.pickle', 'rb'))
        result = list()
        for sent in self.sentences:
            features = self.get_features(sent, postagger)
            feature_values = [features[key] for key in features.keys()]
            feature_values.pop(0)
            result.append(feature_values)
        return result, self.labels

    def load_stop_words(self):
        return ['.', '?', '!', ':']

    def remove_stop_words(self, sentence):
        sent = sentence.replace(',', '')
        for sw in self.stop_words:
            sent = sent.replace(sw, '')
        return sent

tsc = TensorFlowClassifier()
texts, labels = tsc.get_data()

limit = 300
(test_data, test_labels) = (texts[:limit], labels[:limit])
(train_data, train_labels) = (texts[limit:], labels[limit:])

train_labels = keras.utils.to_categorical(train_labels, num_classes=8)
test_labels = keras.utils.to_categorical(test_labels, num_classes=8)

train_data = keras.preprocessing.sequence.pad_sequences(
    train_data, value=0, padding='post', maxlen=15
)
test_data = keras.preprocessing.sequence.pad_sequences(
    test_data, value=0, padding='post', maxlen=15
)

model = keras.Sequential()
model.add(keras.layers.Dense(32, activation='relu', input_dim=15))
model.add(keras.layers.Dense(32, activation=tf.nn.relu))
model.add(keras.layers.Dense(8, activation='softmax'))

model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    train_data,
    train_labels,
    epochs=150
)

a, b = model.evaluate(test_data, test_labels)

print(a)
print(b)
previsao = model.predict(test_data)
teste_matrix = [np.argmax(t) for t in test_labels]
previsoes_matrix = [np.argmax(t) for t in previsao]

confusao = confusion_matrix(teste_matrix, previsoes_matrix)
print(confusao)
import pdb;pdb.set_trace()
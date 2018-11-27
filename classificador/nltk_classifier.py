import nltk
import pickle

def get_data(dataset_path='conversas.txt'):
    results = list()

    arq = open(dataset_path, 'r')
    raw = arq.read().lower()

    lines = raw.split('\n')
    for line in lines:
        splited_line = line.split(',')
        sentence = ','.join(splited_line[:-2])
        try:
            results.append((sentence, int(splited_line[-2])))
        except:
            pass

    return results

def get_features(post, postagger):
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
    err_words = ['problema', 'erro', 'travando', 'funciona', 'funcionando', 'abre', 'não', 'n', 'trava', 'abrindo', 'abriu', 'abrir', 'nao']
    ag_words = ['obrigada', 'obrigado', 'obg']
    saud_words = ['dia', 'tarde', 'noite', 'bom', 'boa', 'oi', 'ola', 'olá', 'oii']
    neg_words = ['não', 'nao', 'n']
    conf_words = ['é', 'sim', 'certo', 'bom', 'pronto', 'ok', 'bem', 'beleza', 'blz', 'sei', 'entendo', 'entendido', 'entendi', 'já', 'ja']
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
    features['question_sign'] = '?' in post
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

postagger = pickle.load(open('postagger.pickle', 'rb'))
dataset = get_data()
fSets = [(get_features(text, postagger), label) for (text, label) in dataset]
size = int(len(fSets) * 0.1)
trainSet, testSet = fSets[size:], fSets[:size]
c1 = nltk.NaiveBayesClassifier.train(trainSet)
print(nltk.classify.accuracy(c1, testSet))

f = open('classifier.pickle', 'wb')
pickle.dump(c1, f)
f.close()

classifier_me = nltk.MaxentClassifier.train(trainSet, max_iter=20)
acuracia = nltk.classify.accuracy(classifier_me, testSet)
print("Acuracia MaxEnt: "+str(acuracia))
# PROBLEMA,1
# AGRADECIMENTO,2
# SAUDACAO,3
# OUTROS,4
# DUVIDA,5
# NEGACAO,6
# CONFIRMACAO,7
# INFORMACAO,8
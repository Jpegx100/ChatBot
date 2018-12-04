import random
import constantes
import tf_classifier

def saudar():
    saudacoes = ['Oi', 'Bom dia', 'Boa tarde', 'Boa noite', 'Olá']
    return random.choice(saudacoes) + ', em que posso lhe ajudar? '

def contra_agradecimento():
    msgs = [
        'foi um prazer lhe ajudar',
        'conte comigo sempre que precisar',
        'qualquer outra coisa é só chamar ;)',
        'precisando estou aqui pra lhe atender'
    ]
    return 'Por nada, ' + random.choice(msgs)

def confirmado():
    return 'Certo'

def negado():
    return 'Vish'

def resolve_problema():
    print('Seu problema foi informado para os técnicos')

def responder(pergunta):
    return 'Deixe pensar a respeito e jajá lhe respondo...'

def adicionar_informacao(info):
    print('Obrigado pela informação, será de grande ajuda na solução do problema')

def nao_entendido():
    return 'Não consegui lhe entender'


tsc = tf_classifier.TensorFlowClassifier()
input_text = ''
saida = 'Entre com o texto:'
while input_text != 'exit':
    input_text = input(saida + ' ')
    classe = tsc.predict(input_text.lower())
    
    if classe == constantes.CLASSE_PROBLEMA:
        resolve_problema()
    elif classe == constantes.CLASSE_AGRADECIMENTO:
        saida = contra_agradecimento()
    elif classe == constantes.CLASSE_SAUDACAO:
        saida = saudar()
    elif classe == constantes.CLASSE_OUTROS:
        saida = nao_entendido()
    elif classe == constantes.CLASSE_DUVIDA:
        saida = responder(input_text.lower())
    elif classe == constantes.CLASSE_NEGACAO:
        saida = negado()
    elif classe == constantes.CLASSE_CONFIRMACAO:
        saida = confirmado()
    elif classe == constantes.CLASSE_INFORMACAO:
        adicionar_informacao(input_text.lower())
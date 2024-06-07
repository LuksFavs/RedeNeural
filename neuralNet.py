import pandas as pd
import numpy as np
import math as m

iris = pd.read_csv('iris.csv')

count = {}
for j in  range(len(iris)):
    if iris['target'][j] in count:
        count[iris['target'][j]].append(j)
    else:
        count[iris['target'][j]] = [j]

def stratKFold(k, classindex, dataLen):
    ## k = numero de folds
    ## classindex = dicionario com os indexes das classes
    ## dataLen = tamanho da database
    quantidadeClasses = len(classindex)
    rk= range(k)
    over = 0
    folds =[[] for _ in rk]
    while over<dataLen:
        for i in rk:
            for j in classindex:
                if len(classindex[j]) == 0:
                    continue
                over+=1
                aux = classindex[j].pop(0)
                folds[i].append(aux)

    folds = stratKFold(5,count,150)

class ativacao:
    def sigmoide(z):
        return 1/(1+np.exp(-z))
    def tanh(z):
        return (np.exp(z)-np.exp(-z)) / (np.exp(z)+np.exp(-z))
    def relu(z):
        z[z<0] = 0
        return z
    def leakyRelu(z):
        return np.maximum(z,z*0.01) 

class perda:
    def m2(est, resp):
        return np.power(resp-est, 2)
    def cE(est, resp):
        return -(resp*np.log(est)+(1-resp)*np.log(esp))
    def cce(est, resp):
        return np.sum(resp*np.log(esp))

def redeN(camadas, ativacao, perda):
    # camadas = array onde len(camadas) = quantidade de camadas e
    # camadas[i] = quantidade de neuronios na camada i
    # ativacao = funcao ativacao a ser usada, algumas disponiveis na classe ativacao acima
    # perda = funcao de perda a ser usada, algumas disponiveis na classe perda acima
    return


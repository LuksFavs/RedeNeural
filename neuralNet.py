import pandas as pd
import numpy as np
import math as m
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler #use .transform to transform

#TODO - backpropagation, lossfunction on main loop, implement minmaxScaler and standard

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
    def sigmoide(self, z):
        aux = z.copy()
        return 1/(1+np.exp(-1 * aux))
    def tanh(self, z):
        aux = z.copy()
        return (np.exp(aux)-np.exp(-aux)) / (np.exp(aux)+np.exp(-aux))
    def relu(self, z):
        aux = z.copy()
        aux[aux<0] = 0
        return aux
    def leakyRelu(self, z):
        aux = z.copy()
        return np.maximum(aux,aux*0.01) 

class perda:
    def m2(self, est, resp):
        return np.power(resp-est, 2)
    def cE(self, est, resp):
        return -(resp*np.log(est)+(1-resp)*np.log(est))
    def cce(self, est, resp):
        return np.sum(resp*np.log(est))

def redeN(inicial, final, ativFunc, perda, dataset, treinamento, teste):
    # camadas = array onde len(camadas) = quantidade de camadas e
    # camadas[i] = quantidade de neuronios na camada i
    # ativacao = funcao ativacao a ser usada, algumas disponiveis na classe ativacao acima
    # perda = funcao de perda a ser usada, algumas disponiveis na classe perda acima
    # 1 camada escondida com (inicial + final) / 2 neuroneos

    tamanhoInter = np.random.randint(inicial*1.5, inicial*2)
    tamanhoTot = tamanhoInter + inicial + final
    pesosInt = np.random.normal( size = (inicial, tamanhoInter) )
    pesosFin = np.random.normal( size = (tamanhoInter, final) )
    bias = np.zeros(shape=2)
    entrada = np.random.uniform(0,2,size=4) 
    for i in range(30):
        z1=np.matmul(entrada, pesosInt)
        z1=z1+bias[0]
        a1=ativFunc(z1)
        z2=np.matmul(a1,pesosFin)
        z2=z2+bias[1]
        a2=ativFunc(z2)

    return

classe = ativacao()
ati = classe.sigmoide
classe2 = perda()
per=classe2.m2

classif = iris["target"]
iris = iris.drop(["target"], axis=1)
entradas = iris.to_numpy()
onehot = pd.get_dummies(classif, dtype=float)
onehot=onehot.to_numpy()
print(entradas)

# redeN(4,3,ati,per,,7,4)

# q = np.random.uniform(-2,2,20)
# print("base:\n", q)
# ati = ativacao()
# sig = ati.sigmoide(z = q)
# tanh = ati.tanh(z=q)
# re = ati.relu(z=q)
# lre = ati.leakyRelu(z=q)
#
# per = perda()
# m = per.m2(sig, q)
# c = per.cE(sig, q)
# cc = per.cce(sig, q)
#
# print("ativacao:\n", sig)
# print(tanh)
# print(re)
# print(lre)
# print("base:\n", q)
# print("perda:\n")
# print(m)
# print(c)
# print(cc)



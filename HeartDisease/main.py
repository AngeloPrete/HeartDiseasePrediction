import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
from sklearn.cluster import KMeans
from pgmpy.inference import VariableElimination
from ApprendimentoSupervisionato import addestra
from ApprendimentoNonSupervisionato import getK_regolaGomito
from BayesianNetwork import stampaReteBayesiana, apprendi_rete_bayesiana
from RagionamentoRelazionale import aggiungi_fatti, aggiungi_regole, inferisci_feature
from pyswip import Prolog
import os
from imblearn.over_sampling import SMOTE

# Ottieni il percorso del file corrente
current_dir = os.path.dirname(os.path.realpath(__file__))

# Crea il percorso del file 'heart.csv' relativo al file corrente
dataset_path = os.path.join(current_dir, 'dataset', 'heart.csv')

data = pd.read_csv(f"{dataset_path}")

dataSet = data.copy()

#APPRENDIMENTO SUPERVISIONATO

#Trovo le variabili categoriche
categorical_columns = dataSet.select_dtypes(include=['object', 'category']).columns

# Trasformo le variabili categoriche con il one-hot encoding
dataSet = pd.get_dummies(dataSet, columns=categorical_columns)

# Normalizzo tutte le varibili numeriche continue
scaler = MinMaxScaler()
dataSet = pd.DataFrame(scaler.fit_transform(dataSet), columns=dataSet.columns)

X = dataSet.drop(columns=['HeartDisease'])  
y = dataSet['HeartDisease'] 

# Verifico se le classi sono distribuiti in modo equo
counts = y.value_counts()

plt.figure(figsize=(7, 7))
plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])
plt.title('Distribuzione delle Classi')
plt.axis('equal')  # Per fare in modo che il grafico a torta sia circolare
plt.show()

#Divido il dataset in dati di training e test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Addestro e valuto il modello Randomforest
addestra(X_train, X_test, y_train, y_test,X,y,RandomForestClassifier(random_state=42))

#Addestro e valuto il modello Decision Tree
addestra(X_train, X_test, y_train, y_test,X,y,DecisionTreeClassifier(random_state=42))

#Addestro e valuto il modello di regressione logistica
addestra(X_train, X_test, y_train, y_test,X,y,LogisticRegression(max_iter=10000, random_state=42))


#APPRENDIMENTO NON SUPERVISIONATO

'''Il kmeans viene eseguito solo sulle feature numeriche continue'''

dataSet_copia = data.drop(columns=categorical_columns)
dataSet_copia = dataSet_copia.drop(columns=['HeartDisease','FastingBS'])
dataSet_copia = pd.DataFrame(scaler.fit_transform(dataSet_copia), columns = dataSet_copia.columns)

k = getK_regolaGomito(dataSet_copia)

# Eseguo il clustering K-means
kmeans = KMeans(n_clusters=k, random_state=42)
dataSet_copia['Cluster'] = kmeans.fit_predict(dataSet_copia)

'''rieseguo i modelli di apprendimento supervisionato con la nuova feature Cluster'''

# Aggiungo la feature Cluster al dataset precedente
dataSet['Cluster']=dataSet_copia['Cluster']

X = dataSet.drop(columns=['HeartDisease'])  
y = dataSet['HeartDisease'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

addestra(X_train, X_test, y_train, y_test,X,y,RandomForestClassifier(random_state=42))

addestra(X_train, X_test, y_train, y_test,X,y,DecisionTreeClassifier(random_state=42))

addestra(X_train, X_test, y_train, y_test,X,y,LogisticRegression(max_iter=10000, random_state=42))

#INFERENZA NUOVE FEATURE ATTRAVERSO UNA KNOWLEDGE BASE

#Popolo la KB con i fatti presenti nel dataset

prolog = Prolog()

aggiungi_fatti(data,prolog)

aggiungi_regole(prolog) #Inserisco le regole per inferire nuove feature

inferisci_feature(data,prolog)


#RAGIONAMENTO PROBABILISTICO E RETE BAYESIANA

#Sfruttando le feature ingegnerizzate dalla KB apprendo la rete bayesiana 

kbins = KBinsDiscretizer(n_bins=10,encode='ordinal',strategy='uniform')

data['MaxHR'] = kbins.fit_transform(data[['MaxHR']])

#Tolgo le feature continue
feature = data.drop(columns=['RestingBP', 'Cholesterol', 'Oldpeak'])

bins = [18, 35, 50, 60, 150]
feature['Age'] = pd.cut(feature['Age'],bins=bins,labels=["Giovane","Adulto","50-60","Over 60"], right=False)

model = apprendi_rete_bayesiana(feature)

# Stampo i CPDs associati al modello
for cpd in model.get_cpds():
    print(cpd)

stampaReteBayesiana(model)

# Inizializzol'inferenza
inference = VariableElimination(model)

query_result = inference.query(variables=['HeartDisease'], evidence={
    'Oldpeak_categoria': 'High',
    'ST_Slope': 'Down',
    'Cholesterol_categoria': 'High',
})

print("Query: Distribuzione di probabilità Heart Disease sapendo che Oldpeak_categoria: High, ST_Slope: Down, Cholesterol_categoria:High")
print(query_result)

query_result = inference.query(variables=['HeartDisease'], evidence={
    'Oldpeak_categoria': 'High',
    'ST_Slope': 'Down',
    'Cholesterol_categoria': 'High',
    'Age': '50-60'
})

print("Query: Distribuzione di probabilità Heart Disease sapendo che Oldpeak_categoria: High, ST_Slope: Down, Cholesterol_categoria: High, Age:50-60")
print(query_result)

query_result = inference.query(variables=['HeartDisease'], evidence={
    'Oldpeak_categoria': 'Moderate',
    'ST_Slope': 'Down',
    'Cholesterol_categoria': 'Normal',
    'Age': '50-60'
})

print("Query: Distribuzione di probabilità Heart Disease sapendo che Oldpeak_categoria: Moderate, ST_Slope: Down, Cholesterol_categoria: Normal, Age:50-60")
print(query_result)

query_result = inference.query(variables=['HeartDisease'], evidence={
    'Oldpeak_categoria': 'Moderate',
    'ST_Slope': 'Down',
    'Cholesterol_categoria': 'Normal',
    'Age': 'Giovane'
})

print("Query: Distribuzione di probabilità Heart Disease sapendo che Oldpeak_categoria: Moderate, ST_Slope: Down, Cholesterol_categoria: Normal, Age:Giovane")
print(query_result)
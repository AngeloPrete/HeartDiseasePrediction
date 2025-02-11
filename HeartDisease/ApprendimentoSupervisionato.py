import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import numpy as np 
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score


def getGrid_params(model) :
    if isinstance(model,RandomForestClassifier) :
        param = getGriglia_parametersRandomForest() 
    else :
        if isinstance(model,LogisticRegression) :
            param = getGriglia_parametersLogisticRegression()
        else :
                param = getGriglia_parametersDecisionTree()
    
    return param

# Funzione per addestrare e valutare il modello passato come parametro
def addestra(X_train, X_test, y_train, y_test,X,y,model):
    
    param = getGrid_params(model)

    best_model, best_hyperparameters = getBest_model_and_hyperparameters(model,param,X_train,y_train)
 
    print(f"Migliori parametri del modello {best_hyperparameters}")

    # Definisco la strategia di K-Fold Cross Validation (K=5)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
  
    # Eseguo la K-Fold Cross Validation sull'intero dataset
    scores_accuratezza = cross_val_score(best_model, X, y, cv=kf, scoring='accuracy')
    # Accuratezza media
    print(f"Accuratezza media : {scores_accuratezza.mean():.4f} ± {scores_accuratezza.std():.4f}")

    plot_curva_apprendimento(best_model,X,y)  


def getGriglia_parametersLogisticRegression():
    # Definiamo la griglia di iperparametri
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }
    return param_grid

def getGriglia_parametersDecisionTree():
    # Definiamo la griglia di iperparametri
    param_grid = {
        'criterion': ['entropy','log_loss'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    return param_grid

def getGriglia_parametersRandomForest():
    # Definiamo la griglia di iperparametri
    param_grid = {
        'criterion': ['entropy','log_loss'],
        'n_estimators': [100, 200, 300],  # Numero di alberi
        'max_depth': [None, 5, 10, 20, 30],  # Profondità massima
        'min_samples_split': [5, 10],  # Minimo numero di campioni per dividere un nodo
        'min_samples_leaf': [1, 2, 4],    # Minimo numero di campioni per lasciare una foglia
    }
    return param_grid

def valutaModello(y_test, y_pred):

    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test,y_pred)

    return accuracy,recall,f1

def getBest_model_and_hyperparameters(estimator,param_grid,X_train,y_train):
    # Applichiamo GridSearchCV con 5-fold cross-validation
    grid_search = GridSearchCV(estimator, param_grid, cv=5, n_jobs=-1, scoring='accuracy')

    #Addestriamo il modello
    grid_search.fit(X_train, y_train)

    # Estraiamo il miglior modello da GridSearchCV
    best_model = grid_search.best_estimator_

    return best_model,grid_search.best_params_

def plot_curva_apprendimento(model,X,y):
    # Calcolo della curva di apprendimento
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5)

    # Media
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    # Plot della curva di apprendimento
    plt.plot(train_sizes, train_mean, label="Training Score", marker='o')
    plt.plot(train_sizes, test_mean, label="Validation Score", marker='s')
    plt.xlabel("Dimensione del Training Set")
    plt.ylabel("Accuratezza")
    plt.title("Curva di Apprendimento")
    plt.legend()
    plt.show()

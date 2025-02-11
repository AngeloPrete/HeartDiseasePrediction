import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.estimators import HillClimbSearch, K2Score
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator

def apprendi_rete_bayesiana(feature):
    # Applichiamo l'algoritmo Hill Climbing
    hc = HillClimbSearch(feature)
    best_model = hc.estimate(scoring_method='k2score')

    # Crea la rete bayesiana
    model = BayesianNetwork(best_model)

    # Aggiungi i CPDs stimati tramite Maximum Likelihood Estimation
    model.fit(feature, estimator=MaximumLikelihoodEstimator)

    return model

def stampaReteBayesiana(model):

    G = nx.DiGraph()  # Grafico orientato

    # Aggiungi gli archi dalla rete bayesiana
    G.add_edges_from(model.edges())

    pos = nx.spring_layout(G, k=1.5)  # 'k' controlla la distanza tra i nodi

    # Aumenta la dimensione della figura per migliorare la disposizione
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=12, font_weight="bold")

    # Mostra il grafico

    plt.title("Rete Bayesiana Appresa")
    plt.savefig('bayesian_network.png')
    plt.show()
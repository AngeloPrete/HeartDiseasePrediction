from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator


def getK_regolaGomito(dataSet_copia):
    # Calcola la somma delle distanze per vari valori di k
    inertia = []
    K = range(1, 11)

    for k in K: 
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(dataSet_copia)
        inertia.append(kmeans.inertia_)

    knee_locator = KneeLocator(K, inertia, curve="convex", direction="decreasing")

    # Plot del metodo del gomito
    plt.figure(figsize=(10, 6))
    plt.plot(K, inertia, marker='o', linestyle='-')
    plt.scatter(knee_locator.knee, knee_locator.elbow_y, color='red', s=100, label=f'Knee = {knee_locator.knee}')
    plt.title('Metodo del Gomito per trovare k')
    plt.xlabel('Numero di cluster (k)')
    plt.ylabel('Inertia')
    plt.legend()
    plt.show()

    return knee_locator.knee
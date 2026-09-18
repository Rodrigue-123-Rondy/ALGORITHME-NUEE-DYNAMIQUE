import numpy as np

def distance_euclidienne(a, b):
    return np.sqrt(np.sum((a - b)**2, axis=1))

class NueesDynamiques:
    def __init__(self, k=3, max_iter=100):
        self.k = k
        self.max_iter = max_iter

    def fit(self, X):
        # 1. Initialisation des centres
        indices = np.random.choice(len(X), self.k, replace=False)
        self.centres = X[indices]

        for _ in range(self.max_iter):
            # 2. Affectation
            clusters = [[] for _ in range(self.k)]
            labels = []
            for point in X:
                distances = [np.linalg.norm(point - c) for c in self.centres]
                cluster_idx = np.argmin(distances)
                labels.append(cluster_idx)

            # 3. Mise à jour des centres
            nouveaux_centres = np.array([X[np.array(labels) == i].mean(axis=0) for i in range(self.k)])

            # 4. Condition de convergence
            if np.all(self.centres == nouveaux_centres):
                break
            self.centres = nouveaux_centres

        return np.array(labels)

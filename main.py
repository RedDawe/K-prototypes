rng = range(1, 10)
n_epochs = 30
gamma = 0.3 # THE puzzling parameter (weight between categorical and continuous features)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def do_k_prototypes(K):
    clusters = np.copy(X[0:K])

    for epoch in range(n_epochs):

        cluster_indxs = []
        for _ in range(K):
            cluster_indxs.append([])

        J = 0

        for i, row in enumerate(X):
            differences = []

            for cluster in clusters:

                difference = 0

                for j in range(6):
                    difference += (cluster[j] - row[j]) ** 2

                for j in range(6, 10):
                    difference += 0 if cluster[j] == row[j] else 1 * gamma

                differences.append(difference)

            indx = np.argmin(differences)

            cluster_indxs[indx].append(i)

            J += differences[indx] / X.shape[0]

        
        cluster_list = []
        for i in range(K):
            # clusters[i, :] = np.mean(X[cluster_indxs[i], :], axis=0)

            cluster_list.append(np.mean(X[cluster_indxs[i], :], axis=0))

        clusters = np.stack(cluster_list, axis=0)
        #clusters = np.concatenate([clusters[:, :6], np.round(clusters[:, 6:])], axis=-1)


        for j in range(6, 10):
          for c in range(len(cluster_indxs)):
            categorical_count = {}
            for i in cluster_indxs[c]:
              if X[i, j] in categorical_count:
                categorical_count[X[i, j]] += 1
              else:
                categorical_count[X[i, j]] = 1

            var = -1
            maximum = 0
            for key, value in categorical_count.items():
              if value > maximum:
                var = key
                maximum = value

            clusters[c, j] = var

    return J, cluster_indxs, clusters

workclass = {
    'Private' : 0,
    'Self-emp-not-inc' : 1,
    'Self-emp-inc' : 2,
    'Federal-gov' : 3,
    'Local-gov' : 4,
    'State-gov' : 5,
    'Without-pay' : 6,
    'Never-worked' : 7,
    '?' : 8
}

marital_status = {
    'Married-civ-spouse' : 0,
    'Divorced' : 1,
    'Never-married' : 2,
    'Separated' : 3,
    'Widowed' : 4,
    'Married-spouse-absent' : 5,
    'Married-AF-spouse' : 6,
    '?' : 7
}

sex = {
    'Female' : 0,
    'Male' : 1,
    '?' : 2
}

pay_grade = {
    '>50K\n' : 0,
    '<=50K\n' : 1,
    '?' : 2
}


X = []

with open('adult.data') as f:
    for i in f:
        s = i.replace(' ', '').split(',')

        if len(s) == 15:
            x = []

            x.append(int(s[0]))
            x.append(int(s[2]))
            x.append(int(s[4]))
            x.append(int(s[10]))
            x.append(int(s[11]))
            x.append(int(s[12]))

            x.append(workclass[s[1]])
            x.append(marital_status[s[5]])
            x.append(sex[s[9]])
            x.append(pay_grade[s[14]])

            X.append(x)

X = np.array(X)
np.random.shuffle(X)

#X = X[:500, :]

min = np.amin(X[:, 0:6], 0)
max = np.amax(X[:, 0:6], 0)

X = np.concatenate([(X[:, 0:6] - min) / (max - min), X[:, 6:]], -1)

results = []

for K in rng:
    J, _, __ = do_k_prototypes(K)
    results.append(J)

plt.scatter(rng, results)
plt.show()

K = int(input('Your choice:'))

_, cluster_indxs, clusters = do_k_prototypes(K)

X_embedded = TSNE(n_components=2).fit_transform(X)

for i in range(len(cluster_indxs)):
    plt.scatter(X_embedded[cluster_indxs[i], 0], X_embedded[cluster_indxs[i], 1])
plt.show()

print(clusters)

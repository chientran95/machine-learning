import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def pca_explained_variance_ratio(feats):
    n_comps = min(feats.shape)
    pca = PCA(n_components=n_comps)
    pca.fit(feats)

    # Plot
    plt.plot(range(0, n_comps), pca.explained_variance_ratio_.cumsum())
    plt.ylabel('Explained Variance')
    plt.xlabel('Principal Components')
    plt.show()
    return pca.explained_variance_ratio_

def hammingDist(str1, str2):
    count = 0
    for x, y in zip(str1, str2):
        count += (x != y)
    return count

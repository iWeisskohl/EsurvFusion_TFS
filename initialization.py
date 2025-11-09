import torch
from sklearn.cluster import KMeans

def Prototype_init(X_dict, y, K_dict, nstart=100, c=1):
    prototypes = []
    for key in X_dict:
        X = X_dict[key]
        p = X.shape[1]
        K = K_dict[key]

        clus = KMeans(n_clusters=K, max_iter=5000, n_init=nstart, random_state=0).fit(X)

        Beta = torch.zeros(K, p, dtype=torch.float64)
        alpha = torch.zeros(K, dtype=torch.float64)
        sig = torch.ones(K, dtype=torch.float64)
        W = torch.tensor(clus.cluster_centers_, dtype=torch.float64)
        gam = torch.ones(K, dtype=torch.float64)

        for k in range(K):
            mask = torch.eq(torch.tensor(clus.labels_), k)
            ii = torch.nonzero(mask, as_tuple=True)[0]
            nk = len(ii)
            alpha[k] = torch.mean(y[ii])

            if nk > 1:
                gam[k] = 1 / torch.sqrt(torch.tensor(clus.inertia_) / nk)
                sig[k] = torch.std(y[ii])

        gam *= c
        eta = torch.ones(K) * 2

        init = {'alpha': alpha, 'Beta': Beta, 'sig': sig, 'eta': eta, 'gam': gam, 'W': W}
        prototypes.append(init)

    return prototypes

import numpy as np
import matplotlib.pyplot as plt


class Kmeans(object):

    def __init__(self, K=3, max_iter=20, init='', seed=None):
        self.K = K;
        self.max_iter = max_iter
        self.SSE = []
        self.init = init
        self.centroid = None
        np.random.seed(seed)

    def compute_distance(self, A, B):  # compute distance of numpy array A & B
        return np.sum(np.square(A - B), axis=1)  # euclidean^2

    def rand_centroids(self, X):
        m, num_dims = X.shape
        if (self.init == 'kmeans++'):  #距離已有質心越遠的資料點，被選作新質心
            self.centroids = X[np.random.choice(m, 1), :]  # random choose one centroid
            while (self.centroids.shape[0] < self.K):  # until num centroids = K
                distance = np.zeros((m, self.centroids.shape[0]))  # init distance
                for i in range(self.centroids.shape[0]):  # 計算各點對已存在的質心的距離
                    distance[:, i] = self.compute_distance(X, self.centroids[i, :])
                total_distance = np.sum(np.min(distance, axis=1))  # 各點對於最短質心距離的總合
                prob_list = np.min(distance, axis=1) / total_distance  # 各點最短質心距離 / 總合 = 機率
                self.centroids = np.vstack([self.centroids, X[np.random.choice(m, 1, p=prob_list), :]]) # concat new centroid
        else:
            self.centroids = np.zeros((self.K, num_dims))
            rand_index = np.random.choice(m, self.K, replace=False)  # random choose K node of X
            self.centroids = X[rand_index]
        return self

    def change_centroids(self, X):
        m, num_dims = X.shape
        self.centroids = np.zeros((self.K, num_dims))
        for i in range(self.K):
            index = np.where(self.idx[:] == i)  # 所有屬於第K(i)類的點之index
            self.centroids[i] = np.mean(X[index], axis=0)  # 所有K(i)類的點的(x,y)平均值，也就是新的質心
        return self
		
    def calculate_sse(self, X):
        # calculate SSE
        sse = 0
        for i in range(self.K):
            idx_ = np.where(self.idx[:] == i)
            sse += np.sum(np.square(X[idx_, :] - self.centroids[i, :]))
        self.SSE.append(sse)
        return self

    def fit(self, X):
        self.idx = np.zeros(X.shape[0])  # record every x belong to which centroid
        self.cent_record = {}   # use dictionary to save centroids
        clusterchanged = True   # to recode whether the cluster changed
		
        self.rand_centroids(X)  # initialize K centroids

        for k in range(self.K):
            self.cent_record[str(k)] = []
            self.cent_record[str(k)].append(self.centroids[k, :].tolist())  # append initial centroids

        epochs = 0
        while (epochs < self.max_iter):  # until no changed or max_iter
            # can use while(clusterchanged): , will stop when cluster not changed.
            epochs += 1
            clusterchanged = False

            # compute distance between every x and K centroids
            distance = np.zeros((X.shape[0], self.K))
            for c in range(self.K):
                distance[:, c] = self.compute_distance(X, self.centroids[c, :])

            min_distance_index = np.argmin(distance, axis=1)  # find the closest centroid's index
            if ((min_distance_index != self.idx[:]).any()):   # check if any node change cluster
                clusterchanged = True
            self.idx[:] = min_distance_index  # updata the distance between every data and centroids.

            self.calculate_sse(X)  # saved SSE to list

            # update centroids
            self.change_centroids(X)
            for k in range(self.K):
                self.cent_record[str(k)].append(self.centroids[k, :].tolist())
        return self
		
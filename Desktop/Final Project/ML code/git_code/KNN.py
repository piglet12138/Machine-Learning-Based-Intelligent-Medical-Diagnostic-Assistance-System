import numpy as np


def euclidean_distance(x1, x2):
    #计算欧氏距离
    return np.sqrt(np.sum((x1 - x2)**2))

class KNNClassify:
    def __init__(self,k):  
        self.k = k   

    #将X转换成ndarray类型，如果X已经是ndarray则不进行转换  
    #存储训练数据X和相应的标签y
    def fit(self, X, y):
        self.X = np.asarray(X)  
        self.y = np.asarray(y)  

    #将测试的X转换为ndarray结构  
    def _calculate_distance(self, x):
        # 计算测试样本 x 与训练样本 self.X 之间的欧氏距离
        return np.sqrt(np.sum((x - self.X) ** 2, axis=1))
    
    def _get_k_nearest_indices(self, x):
        # 返回最近的 k 个点的索引
        distances = self._calculate_distance(x)
        return np.argsort(distances)[:self.k]
    
    def predict(self, X):
        #基于其k个最近邻的主要类别，为一组测试点预测类标签
        X = np.asarray(X)
        result = []   
        for x in X:  
            indices = self._get_k_nearest_indices(x)
            count = np.bincount(self.y[indices])
            result.append(count.argmax())  
        return np.asarray(result)

    def predict2(self, X):
        #‘predict’的加权版本，其中每个邻居的贡献由其距离的倒数加权
        X = np.asarray(X)
        result = []   
        for x in X:  
            indices = self._get_k_nearest_indices(x)
            weights = 1 / self._calculate_distance(x)[indices]
            count = np.bincount(self.y[indices], weights=weights)
            result.append(count.argmax())  
        return np.asarray(result)  

    




   
    
    
    
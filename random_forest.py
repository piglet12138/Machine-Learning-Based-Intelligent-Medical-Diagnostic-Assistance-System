import numpy as np
from collections import Counter

#随机森林，决策树基于信息增益

# 熵的计算函数
def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])
# 决策树的节点
class DecisionNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        # 对于内部节点来说
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right

        # 对于叶节点来说
        self.value = value


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=2, m=0.4):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.m = m
    # 训练过程
    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        fea_selec = int(self.m * n_features)
        # 停止判断
        if (depth >= self.max_depth or n_labels == 1 or
                n_samples < self.min_samples_split) or fea_selec < 1:
            leaf_value = self._most_common_label(y)
            return DecisionNode(value=leaf_value)

        # 随机选择特征
        feat_idxs = np.random.choice(n_features, fea_selec, replace=False)

        # 寻找最佳划分
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)

        # 递归创建子树
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return DecisionNode(best_feat, best_thresh, left, right)


    # 得到最常见的标签
    def _most_common_label(self, y):
        counter = Counter(y)
        if len(counter) > 0:
            most_common = counter.most_common(1)[0][0]
            return most_common
        else:
            return 0

    # 最佳划分条件
    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            thresholds.sort()
            if len(thresholds) >= 2:
                Thresholds = [(thresholds[i] + thresholds[i + 1]) / 2 for i in range(len(thresholds) - 1)]
            else:
                Thresholds = thresholds
            for threshold in Thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        # 父与子的熵
        parent_entropy = entropy(y)

        left_idxs, right_idxs = self._split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # 信息增益 = 父熵 - 子熵
        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    # 遍历树
    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

class RandomForest:
    def __init__(self, n_trees=100, min_samples_split=2, max_depth=10, n_feats=None, m= 0.4):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []
        self.m = m
    # 训练过程
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(min_samples_split=self.min_samples_split,
                                max_depth=self.max_depth,
                                m = self.m)
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    # Bootstrap 抽样
    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [self._most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)
    # 得到最常见的标签（用于投票）
    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
import numpy as np


class LogisticRegressionGD(object):
    """
     ロジスティック回帰分類器（勾配降下法）

     パラメータ
     -----------
     eta : float
         学習率
     n_iter : int
         訓練データの訓練回数
     random_state : int
         重みを初期化するための乱数シート

     属性
     -----------
     w_ : 1次元配列
         適合後の重み
     cost_ : リスト
         各エポックでの誤差平方和のコスト関数
     """

    def __init__(self, eta=0.01, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
        訓練データに適合させる
        :param X: {配列のようなデータ構造}, shape = [n_examples, n_features]
            訓練データ
            n_examples は訓練データの個数、n_features は特徴量の個数
        :param y: 配列のようなデータ構造, shape = [n_examples]
            目的変数
        :return: self : Object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            # 活性化関数
            output = self.activation(net_input)
            # 誤差の計算
            errors = (y - output)

            # w_1,...,w_m　の更新
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()

            # コスト関数の計算
            # (ロジスティック回帰のコストを計算)
            cost = -y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output)))
            # コストの格納
            self.cost_.append(cost)

        return self

    def net_input(self, X):
        """総入力を計算"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        """ロジスティックシグモイド活性化関数の出力を計算"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        """１ステップ後のクラスラベルを返す"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
from libsvm.svmutil import *
from KNN import *
from MLP import *
from random_forest import *
from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import accuracy_score

def fitness_svm(xn, n ,pd=2,X_train=None, y_train=None,X_test=None, y_test=None):	#accuracy as fitness
    fitness = []
    for i in range(n):
        model = svm_train(y_train, X_train, f'-c {xn[i,0]} -g {xn[i,1]}') #C：{} g：{}
        p_label, p_acc, p_val = svm_predict(y_test, X_test, model)
        ACC, MSE, SCC = p_acc
        fitness.append(ACC)
    return fitness



def fitness_KNN(xn, n ,pd,X_train, y_train,X_test, y_test):	#accuracy as fitness
    fitness = []
    for i in range(n):
        model = KNNClassify(k = int(xn[(i,0)]) if int(xn[(i,0)]) > 1 else 1)#K:{1:100}
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        accuracy = accuracy_score(y_test, prediction)
        fitness.append(accuracy)
    return fitness


def fitness_log(xn, n ,pd,X_train, y_train,X_test, y_test):	#accuracy as fitness
    fitness = []
    for i in range(n):
        model = LogisticRegression(C= xn[(i,0)]) #C:{0.0001,1000}
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        accuracy = accuracy_score(y_test, prediction)
        fitness.append(accuracy)
    return fitness



def fitness_mlp(xn, n ,pd=5,X_train=None, y_train=None,X_test=None, y_test=None):	#accuracy as fitness
    fitness = []
    for i in range(n):
        model =  nn.Sequential(nn.Flatten(),
                    nn.Linear(X_train.shape[1], 64),
                    nn.ReLU(),
                    nn.Dropout(p=xn[i,0] if xn[i,0] < 1 else 0.8),
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Dropout(p=xn[i,1] if xn[i,1] < 1 else 0.8),
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Dropout(p=xn[i,2] if xn[i,2] < 1 else 0.8),
                    nn.Linear(64, len(np.unique(y_train)))
                    )
        prediction = mlp_train(model , batch_size = int(xn[(i,3)]) , num_epochs = 100 , lr = xn[(i,4)] , X = X_train, y = y_train, X_test= X_test, y_test= y_test)
        accuracy = accuracy_score(y_test, prediction)
        fitness.append(accuracy)
    return fitness

def fitness_cart(xn, n ,pd,X_train, y_train,X_test, y_test):	#accuracy as fitness
    fitness = []
    for i in range(n):
        if xn[(i,1)] > 1: xn[(i,1)] =1
        if xn[(i,1)] < 0.5 : xn[(i,1)] =0.5
        model = DecisionTree(max_depth= xn[(i,0)] if xn[(i,0)] > 1 else 1 ,m = xn[(i,1)] )#max_depth:{1,20}, m:{0.5,1}
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        accuracy = accuracy_score(y_test, prediction)
        fitness.append(accuracy)
    return fitness

def fitness_RF(xn, n ,pd,X_train, y_train,X_test, y_test):	#accuracy as fitness
    fitness = []
    for i in range(n):
        model = RandomForest(n_trees= int(xn[(i,0)]) if xn[(i,0)] > 20 else 20 ,max_depth=xn[(i,1)] if xn[(i,1)] > 1 else 1 )#n_tree:{20:200}:{}max_depth:{1,20},
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        accuracy = accuracy_score(y_test, prediction)
        fitness.append(accuracy)
    return fitness

def fitness_XGboost(xn, n ,pd,X_train, y_train,X_test, y_test):	#accuracy as fitness best param is[109.86602684   5.22455799   1.30714474]
    fitness = []
    for i in range(n):
        model = XGBClassifier(n_estimators = int(xn[(i,0)]),max_depth =int(xn[(i,1)]) if xn[(i,1)] > 3 else 3, gamma =xn[(i,2)])# n_estimators:{50:500},max_depth:{3:10},gamma:{0:5}
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        accuracy = accuracy_score(y_test, prediction)
        fitness.append(accuracy)
    return fitness

class CrowSearchAlgorithm:
    def __init__(self, problem_dim=2, population_size=20, awareness_prob=0.1, flight_length=2,lower_bound=[], upper_bound=[], max_iter=100):
        self.pd = problem_dim
        self.n = population_size
        self.ap = awareness_prob
        self.fl = flight_length
        self.l = lower_bound
        self.u = upper_bound
        self.tmax = max_iter

        # Population and Memory initialization
        self.x = np.array(self.init_population())
        self.mem = self.x.copy()
        self.ffit = np.array([])


    def init_population(self):  # init the matrix problem
        x = []
        for i in range(self.n):
            x.append([])
            for j in range(self.pd):
                x[i].append(self.l[j] - (self.l[j] - self.u[j]) * (random.random()))
        return x





    def optimize(self,fitness,data, X_train, y_train, X_test, y_test):
        '''
        :param fitness: fitness_function of target model
        :param data: dataset working on
        :return: best meat-param
        '''
        self.X_train, self.y_train, self.X_test, self.y_test = X_train, y_train, X_test, y_test
        self.fit_mem = np.array(fitness(self.x,self.n, 2, self.X_train, self.y_train, self.X_test, self.y_test))
        for t in range(self.tmax):

            num = np.array([random.randint(0, self.n - 1) for _ in range(self.n)])  # Generation of random candidate crows for following (chasing)
            xnew = np.empty((self.n, self.pd))
            for i in range(self.n):
                if (random.random() > self.ap):
                    for j in range(self.pd):
                        xnew[(i, j)] = self.x[(i, j)] + self.fl * ((random.random()) * (self.mem[(num[i], j)] - self.x[(i, j)]))
                        if xnew[(i, j)] < 0:  # gamma > 0
                            xnew[(i, j)] = self.u[j] + xnew[(i, j)] % self.u[j]
                else:
                    for j in range(self.pd):
                        xnew[(i, j)] = self.l[j] - (self.l[j] - self.u[j]) * random.random()
            xn = xnew.copy()
            ft = np.array(fitness(xn, self.n, self.pd, self.X_train, self.y_train, self.X_test,
                                     self.y_test))  # Function for fitness evaluation of new solutions

            # Update position and memory#
            for i in range(self.n):
                if (xnew[i] > self.l).all() and (xnew[i] < self.u).all():
                    self.x[i] = xnew[i].copy()  # Update position
                    if (ft[i] > self.fit_mem[i]):
                        self.mem[i] = xnew[i].copy()  # Update memory
                        self.fit_mem[i] = ft[i]
            self.ffit = np.append(self.ffit, np.amax(self.fit_mem))  # Best found value until iteration t
            ngbest, = np.where(np.isclose(self.fit_mem, max(self.fit_mem)))
            if self.pd ==2:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(self.mem.T[0], self.mem.T[1], self.fit_mem)
                plt.title(f'{fitness.__name__}：meta_parm & accurancy\n')
                plt.show()
                #plt.savefig(f'{fitness.__name__},{data}.png', dpi=300)
            if self.pd ==1:
                fig = plt.figure()
                plt.title(f'{fitness.__name__}：meta_parm & accurancy\n')
                plt.scatter([int(i) for i in self.mem], [float(i) for i in self.fit_mem])
                plt.show()
                #plt.savefig(f'{fitness.__name__},{data}.png', dpi=300)
            return self.mem[ngbest[0]]


from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtUiTools import QUiLoader
from Mutilayer_perceptron import *
from utilities import Data_Set
from CSA import *
from libsvm.svmutil import *
from sklearn.model_selection import train_test_split
import os
import sys
from libsvm.svmutil import *
from KNN import *
from MLP import *
from random_forest import *
from sklearn.linear_model import LogisticRegression



def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)

    return os.path.join(os.path.abspath("."), relative_path)


# 使用 resource_path 函数来获取正确的文件路径
ui_file_path = resource_path('system ui.ui')






class system:

    def __init__(self):
        # 从文件中加载UI定义

        # 从 UI 定义中动态 创建一个相应的窗口对象
        # 注意：里面的控件对象也成为窗口对象的属性了
        # 比如 self.ui.button , self.ui.textEdit
        self.ui = QUiLoader().load(ui_file_path)
        self.disease = 'liver disease'
        self.method = 'CART'
        self.input = None
        self.sample = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_train = None
        #储存模型的字典
        self.model_dictionary = {}
        self.dataset = Data_Set()

        self.ui.button1.clicked.connect(self.handbutton1)
        self.ui.combobox_disease.currentIndexChanged.connect(self.handcombomox_disease_indexchange)
        self.ui.combobox_method.currentIndexChanged.connect(self.handcombomox_method_indexchange)
        self.ui.input_text.textChanged.connect(self.handtextchange)


    def handtextchange(self):
        self.input = self.ui.input_text.toPlainText()
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setText(self.input)
        msgBox.setWindowTitle("input")
        msgBox.setStandardButtons(QMessageBox.Ok)
        #msgBox.exec()

    def handcombomox_disease_indexchange(self):

        self.disease = self.ui.combobox_disease.currentText()
        self.X_train,self.y_train, self.X_test, self.y_test = self.dataset.load_data(self.disease)
        ranint = random.randint(0, len(self.y_test)-1)
        self.ui.test_sample.clear()
        self.ui.test_sample.append('you can try this sample:\n')
        self.ui.test_sample.append(f'{self.X_test[ranint]}\n')
        self.ui.test_sample.append(f'the label is {self.y_test[ranint]}')

        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setText("Current disease is " + self.disease)
        msgBox.setWindowTitle("Current disease")
        msgBox.setStandardButtons(QMessageBox.Ok)
        msgBox.exec()


    def handcombomox_method_indexchange(self):
        self.method = self.ui.combobox_method.currentText()
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setText("Current method is " + self.method)
        msgBox.setWindowTitle("Current method")
        msgBox.setStandardButtons(QMessageBox.Ok)
        msgBox.exec()

    def handbutton1(self):
        #将输入文本转化为arrey
        input = self.input
        item = input.split()
        self.sample = [float(p) for p in item]
        self.sample = np.array(self.sample)
        self.sample = np.expand_dims(self.sample, axis=0)


        #导入数据集
        data = self.disease
        self.X_train,self.y_train, self.X_test, self.y_test = self.dataset.load_data(data)
        #检查输入格式
        if len(self.sample[0]) != len(self.X_train[0]):
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Information)
            msgBox.setText(f"please input sample with {len(self.X_train[0])} features")
            msgBox.setWindowTitle("notice")
            msgBox.setStandardButtons(QMessageBox.Ok)
            msgBox.exec()
            return
#导入模型
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setText(f"{self.method} is running.")
        msgBox.setWindowTitle("notice")
        msgBox.setStandardButtons(QMessageBox.Ok)
        msgBox.exec()

        if self.method == "CART":
            if self.disease + ':' + self.method not in self.model_dictionary:
                cso_cart = CrowSearchAlgorithm(problem_dim=2, population_size=20, awareness_prob=0.1, flight_length=2,
                                               lower_bound=[1, 0.8], upper_bound=[20, 1], max_iter=20)
                par = cso_cart.optimize(fitness_cart, data, self.X_train, self.y_train, self.X_test, self.y_test)
                cart = DecisionTree(max_depth=par[0], m=par[1])
                cart.fit(self.X_train, self.y_train)
                prediction = cart.predict(self.X_test)
                acc = np.sum(prediction == self.y_test) / len(self.y_test)
                self.model_dictionary[self.disease + ':' + self.method] = {'model':cart,'acc':acc}

        if self.method == 'Random forest':
            if self.disease + ':' + self.method not in self.model_dictionary:
                cso_RF = CrowSearchAlgorithm(problem_dim=3, population_size=5, awareness_prob=0.1, flight_length=2,
                                             lower_bound=[80, 4, 0.3], upper_bound=[120, 12, 1], max_iter=3)
                par = cso_RF.optimize(fitness_RF, data, self.X_train, self.y_train, self.X_test, self.y_test)
                forest = RandomForest(n_trees=int(par[0]), max_depth=par[1], m=par[2])
                forest.fit(self.X_train, self.y_train)
                prediction = forest.predict(self.X_test)
                acc = np.sum(prediction == self.y_test) / len(self.y_test)
                self.model_dictionary[self.disease + ':' + self.method] = {'model':forest,'acc':acc}

        if self.method == 'Logistic regression':
            if self.disease + ':' + self.method not in self.model_dictionary:
                cso_log = CrowSearchAlgorithm(problem_dim=1, population_size=20, awareness_prob=0.1, flight_length=2,
                                              lower_bound=[0.0001], upper_bound=[1.0], max_iter=50)
                par = cso_log.optimize(fitness_log, data,self.X_train, self.y_train, self.X_test, self.y_test)
                log_reg = LogisticRegression(C=par[0])
                log_reg.fit(self.X_train, self.y_train)
                prediction = log_reg.predict(self.X_test)
                acc = np.sum(prediction == self.y_test) / len(self.y_test)
                self.model_dictionary[self.disease + ':' + self.method] = {'model':log_reg,'acc':acc}

        if self.method == 'Mutilayer perceptron':
            if self.disease + ':' + self.method not in self.model_dictionary:
                cso_mlp = CrowSearchAlgorithm(problem_dim=5, population_size=4, awareness_prob=0.1, flight_length=2,
                                              lower_bound=[0.1, 0.1, 0.1, 16, 0.0001],
                                              upper_bound=[0.5, 0.5, 0.5, 512, 0.1], max_iter=4)
                par = cso_mlp.optimize(fitness_mlp, data,self.X_train, self.y_train, self.X_test, self.y_test)

                mlp = nn.Sequential(nn.Flatten(),
                                    nn.Linear(self.X_train.shape[1], 64),
                                    nn.ReLU(),
                                    nn.Dropout(p=par[0]),
                                    nn.Linear(64, 64),
                                    nn.ReLU(),
                                    nn.Dropout(p=par[1]),
                                    nn.Linear(64, 64),
                                    nn.ReLU(),
                                    nn.Dropout(p=par[2]),
                                    nn.Linear(64, len(np.unique(self.y_train)))
                                    )

                prediction = mlp_train(mlp, batch_size=int(par[3]), num_epochs=700, lr=par[4], X=self.X_train, y=self.y_train,
                                       X_test=self.X_test, y_test=self.y_test)
                acc = np.sum(prediction == self.y_test) / len(self.y_test)
                self.model_dictionary[self.disease + ':' + self.method] = {'model':mlp,'acc':acc}

        if self.method == 'KNN':
            if self.disease + ':' + self.method not in self.model_dictionary:
                cso_KNN = CrowSearchAlgorithm(problem_dim=1, population_size=20, awareness_prob=0.1, flight_length=2,
                                              lower_bound=[1], upper_bound=[100], max_iter=50)
                par = cso_KNN.optimize(fitness_KNN, data,self.X_train, self.y_train, self.X_test, self.y_test)
                knn = KNNClassify(k=int(par[0]))  # k = 20
                knn.fit(self.X_train, self.y_train)
                prediction = knn.predict(self.X_test)
                acc = np.sum(prediction == self.y_test) / len(self.y_test)
                self.model_dictionary[self.disease + ':' + self.method] = {'model':knn,'acc':acc}


        if self.method == 'SVM':
            if self.disease + ':' + self.method not in self.model_dictionary:

                cso_svm = CrowSearchAlgorithm(problem_dim=2, population_size=20, awareness_prob=0.1, flight_length=2,
                                              lower_bound=[0.001, 0.0001], upper_bound=[10, 1], max_iter=20)
                par = cso_svm.optimize(fitness_svm, data,self.X_train, self.y_train, self.X_test, self.y_test)

                svm = svm_train(self.y_train, self.X_train, f'-c {par[0]} -g {par[1]}')
                prediction, p_acc, p_val = svm_predict(self.y_test, self.X_test, svm)
                acc = np.sum(prediction == self.y_test) / len(self.y_test)
                self.model_dictionary[self.disease + ':' + self.method] = {'model':svm,'acc':acc}

        if self.method == 'XGBoost':
            if self.disease + ':' + self.method not in self.model_dictionary:
                cso_XGBoost = CrowSearchAlgorithm(problem_dim=3, population_size=10, awareness_prob=0.1,
                                                  flight_length=2, lower_bound=[50, 3, 0], upper_bound=[500, 10, 5],
                                                  max_iter=20)
                par = cso_XGBoost.optimize(fitness_XGboost, data,self.X_train, self.y_train, self.X_test, self.y_test)

                xgboost = XGBClassifier(n_estimators=int(par[0]), max_depth=int(par[1]), gamma=par[2])
                xgboost.fit(self.X_train, self.y_train)
                prediction = xgboost.predict(self.X_test)
                acc = np.sum(prediction == self.y_test) / len(self.y_test)
                self.model_dictionary[self.disease + ':' + self.method] = {'model':xgboost,'acc':acc}

        if self.method == 'STACKING MODEL':
            if self.disease + ':' + self.method not in self.model_dictionary:
                stacking = []
                cso_cart = CrowSearchAlgorithm(problem_dim=2, population_size=20, awareness_prob=0.1, flight_length=2,
                                               lower_bound=[1, 0.8], upper_bound=[20, 1], max_iter=20)
                par = cso_cart.optimize(fitness_cart, data, self.X_train, self.y_train, self.X_test, self.y_test)
                cart = DecisionTree(max_depth=par[0], m=par[1])
                cart.fit(self.X_train, self.y_train)

                self.ui.output.clear()
                self.ui.output.append('cart finished')
                prediction = cart.predict(self.X_test)
                acc = np.sum(prediction == self.y_test) / len(self.y_test)
                self.model_dictionary[self.disease + ':' + 'CART'] = {'model':cart,'acc':acc}


                cso_RF = CrowSearchAlgorithm(problem_dim=3, population_size=5, awareness_prob=0.1, flight_length=2,
                                             lower_bound=[80, 4, 0.3], upper_bound=[120, 12, 1], max_iter=3)
                par = cso_RF.optimize(fitness_RF, data, self.X_train, self.y_train, self.X_test, self.y_test)
                forest = RandomForest(n_trees=int(par[0]), max_depth=par[1], m=par[2])
                forest.fit(self.X_train, self.y_train)

                self.ui.output.append('random forest finished')
                prediction = forest.predict(self.X_test)
                acc = np.sum(prediction == self.y_test) / len(self.y_test)
                self.model_dictionary[self.disease + ':' + 'Random forest'] = {'model':forest,'acc':acc}

                cso_log = CrowSearchAlgorithm(problem_dim=1, population_size=20, awareness_prob=0.1, flight_length=2,
                                              lower_bound=[0.0001], upper_bound=[1.0], max_iter=50)
                par = cso_log.optimize(fitness_log, data,self.X_train, self.y_train, self.X_test, self.y_test)
                log_reg = LogisticRegression(C=par[0])
                log_reg.fit(self.X_train, self.y_train)

                self.ui.output.append('logistic regression finished')
                prediction = log_reg.predict(self.X_test)
                acc = np.sum(prediction == self.y_test) / len(self.y_test)
                self.model_dictionary[self.disease + ':' + 'Logistic regression'] = {'model':log_reg,'acc':acc}

                cso_mlp = CrowSearchAlgorithm(problem_dim=5, population_size=4, awareness_prob=0.1, flight_length=2,
                                              lower_bound=[0.1, 0.1, 0.1, 16, 0.0001],
                                              upper_bound=[0.5, 0.5, 0.5, 512, 0.1], max_iter=4)
                par = cso_mlp.optimize(fitness_mlp, data,self.X_train, self.y_train, self.X_test, self.y_test)

                mlp = nn.Sequential(nn.Flatten(),
                                    nn.Linear(self.X_train.shape[1], 64),
                                    nn.ReLU(),
                                    nn.Dropout(p=par[0]),
                                    nn.Linear(64, 64),
                                    nn.ReLU(),
                                    nn.Dropout(p=par[1]),
                                    nn.Linear(64, 64),
                                    nn.ReLU(),
                                    nn.Dropout(p=par[2]),
                                    nn.Linear(64, len(np.unique(self.y_train)))
                                    )
                stacking.append(mlp_train(mlp, batch_size=int(par[3]), num_epochs=500, lr=par[4], X=self.X_train, y=self.y_train,
                                       X_test=self.X_test, y_test=self.y_test))

                self.ui.output.append('mlp finished')
                prediction = mlp_train(mlp, batch_size=int(par[3]), num_epochs=700, lr=par[4], X=self.X_train,
                                       y=self.y_train,X_test=self.X_test, y_test=self.y_test)
                acc = np.sum(prediction == self.y_test) / len(self.y_test)
                self.model_dictionary[self.disease + ':' + 'Mutilayer perceptron'] = {'model':mlp,'acc':acc}

                cso_KNN = CrowSearchAlgorithm(problem_dim=1, population_size=20, awareness_prob=0.1, flight_length=2,
                                              lower_bound=[1], upper_bound=[100], max_iter=50)
                par = cso_KNN.optimize(fitness_KNN, data, self.X_train, self.y_train, self.X_test, self.y_test)
                knn = KNNClassify(k=int(par[0]))  # k = 20
                knn.fit(self.X_train, self.y_train)

                self.ui.output.append('KNN finished')
                prediction = knn.predict(self.X_test)
                acc = np.sum(prediction == self.y_test) / len(self.y_test)
                self.model_dictionary[self.disease + ':' + 'KNN'] = {'model':knn,'acc':acc}

                cso_svm = CrowSearchAlgorithm(problem_dim=2, population_size=20, awareness_prob=0.1, flight_length=2,
                                              lower_bound=[0.001, 0.0001], upper_bound=[10, 1], max_iter=20)
                par = cso_svm.optimize(fitness_svm, data,self.X_train, self.y_train, self.X_test, self.y_test)

                svm = svm_train(self.y_train, self.X_train, f'-c {par[0]} -g {par[1]}')
                svm_prediction, p_acc, p_val = svm_predict(self.y_test, self.X_test, svm)

                self.ui.output.append('SVM finished')
                prediction = svm_prediction
                acc = np.sum(prediction == self.y_test) / len(self.y_test)
                self.model_dictionary[self.disease + ':' + 'SVM'] = {'model':svm,'acc':acc}

                cso_XGBoost = CrowSearchAlgorithm(problem_dim=3, population_size=10, awareness_prob=0.1,
                                                  flight_length=2, lower_bound=[50, 3, 0], upper_bound=[500, 10, 5],
                                                  max_iter=20)
                par = cso_XGBoost.optimize(fitness_XGboost, data, self.X_train, self.y_train, self.X_test, self.y_test)

                xgboost = XGBClassifier(n_estimators=int(par[0]), max_depth=int(par[1]), gamma=par[2])
                xgboost.fit(self.X_train, self.y_train)

                self.ui.output.append('XGBoost finished')
                prediction = xgboost.predict(self.X_test)
                acc = np.sum(prediction == self.y_test) / len(self.y_test)
                self.model_dictionary[self.disease + ':' + 'XGBoost'] = {'model':xgboost,'acc':acc}


                stacking.append(cart.predict(self.X_test))
                stacking.append(forest.predict(self.X_test))
                stacking.append(log_reg.predict(self.X_test))
                stacking.append(knn.predict(self.X_test))
                stacking.append(svm_prediction)
                stacking.append(xgboost.predict(self.X_test))
                stacking.append(self.y_test)
                stacking = list(map(list, zip(*stacking)))
                stacking = pd.DataFrame(stacking)
                train, test = train_test_split(stacking, test_size=0.2)
                X_train = np.array(train.drop(stacking.columns[-1], axis=1)).astype(float)
                y_train = np.array(train[stacking.columns[-1]])
                X_test = np.array(test.drop(stacking.columns[-1], axis=1)).astype(float)
                y_test = np.array(test[stacking.columns[-1]])

                st_cart = CrowSearchAlgorithm(problem_dim=2, population_size=20, awareness_prob=0.1, flight_length=2,
                                               lower_bound=[1, 0.8], upper_bound=[20, 1], max_iter=20)
                par = st_cart.optimize(fitness_cart, data, X_train, y_train, X_test, y_test)

                st_cart = DecisionTree(max_depth=par[0], m=par[1])
                st_cart.fit(X_train, y_train)
                self.ui.output.append('STACKING model finished')
                prediction = st_cart.predict(X_test)
                acc = np.sum(prediction == y_test) / len(y_test)
                self.model_dictionary[self.disease + ':' + self.method] = {'model':st_cart,'acc':acc}
                self.stacking_models = [cart, forest, log_reg, mlp, knn, svm, xgboost]

        if self.method == 'Mutilayer perceptron':
            prediction = self.model_dictionary[self.disease + ':' + self.method]['model'](torch.tensor(self.sample).type(torch.float32)).detach().numpy().argmax(axis=1)
        elif self.method == 'STACKING MODEL':
            svm_pre ,p_acc, p_val= svm_predict(np.expand_dims(1, axis=0),self.sample,self.stacking_models[5])
            st_sample = [self.stacking_models[3](torch.tensor(self.sample).type(torch.float32)).detach().numpy().argmax(axis=1),
                         self.stacking_models[0].predict(self.sample),self.stacking_models[1].predict(self.sample),self.stacking_models[2].predict(self.sample)
                         ,self.stacking_models[4].predict(self.sample),
                        svm_pre,self.stacking_models[6].predict(self.sample)]
            print(st_sample)
            prediction = self.model_dictionary[self.disease + ':' + self.method]['model'].predict(np.expand_dims(st_sample,axis=0))
        elif self.method == 'SVM':
            prediction, p_acc, p_val = svm_predict(np.expand_dims(1, axis=0),self.sample,self.model_dictionary[self.disease + ':' + self.method]['model'])
        else:
            prediction = self.model_dictionary[self.disease + ':' + self.method]['model'].predict(self.sample)
        acc = self.model_dictionary[self.disease + ':' + self.method]['acc']
        self.ui.output.clear()
        self.ui.output.append('the sample is\n ' + str(self.sample[0]) + '\n')
        self.ui.output.append(f'the prediction of {self.method} is ' + str(int(prediction[0]))+ f'  with accurancy of {acc:.3f}')
        print(self.model_dictionary)




        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setText("run successfully.")
        msgBox.setWindowTitle("notice")
        msgBox.setStandardButtons(QMessageBox.Ok)
        msgBox.exec()













app = QApplication([])
system = system()
system.ui.show()
app.exec_()

# Pyinstaller -D --add-data "dataset:." --hidden-import=libsvm.svmutil --hidden-import=xgboost DA_System.py
# Pyinstaller -D --add-data "dataset:." -D --add-binary "C:\Users\87975\Desktop\Final Project\ML code\pythonProject1\.venv\Lib\site-packages\libsvm\libsvm.dll;." DA_System.py

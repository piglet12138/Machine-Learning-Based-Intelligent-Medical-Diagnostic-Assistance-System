from CSA import *
from libsvm.svmutil import *
from utilities import Data_Set


data_set = Data_Set()
data = 'liver_disease' #heart_disease''breast_cancer''dermatology''liver_disease' 'diabetes1''diabetes2'
X_train,y_train, X_test, y_test = data_set.load_data(data)


#stacking
stacking = []
par =[]
acc = {}

#CART from random forest
cso_cart = CrowSearchAlgorithm(problem_dim=2, population_size=20, awareness_prob=0.1, flight_length=2,lower_bound=[1,0.8], upper_bound=[20,1], max_iter=20)
par= cso_cart.optimize(fitness_cart,data,X_train,y_train,X_test, y_test)

cart = DecisionTree(max_depth= par[0],m = par[1])
cart.fit(X_train,y_train)
prediction = cart.predict(X_test)
pre = cart.predict(X_train)

cm = confusion_matrix(prediction, y_test)
print("预测准确率为：",np.sum(prediction==y_test)/len(y_test))
acc['CART'] = np.sum(prediction==y_test)/len(y_test)
stacking.append(prediction)
#random forest

cso_RF = CrowSearchAlgorithm(problem_dim=3, population_size=5, awareness_prob=0.1, flight_length=2,lower_bound=[80,4,0.3], upper_bound=[120,12,1], max_iter=3)
par = cso_RF.optimize(fitness_RF,data,X_train,y_train, X_test, y_test)

forest = RandomForest(n_trees=int(par[0]),max_depth=par[1], m =par[2])
forest.fit(X_train,y_train)
prediction = forest.predict(X_test)
pre = forest.predict(X_train)

cm = confusion_matrix(prediction, y_test)
print("预测准确率为：",np.sum(prediction==y_test)/len(y_test))
acc['RF'] = np.sum(prediction==y_test)/len(y_test)
stacking.append(prediction)


#logstic_regression
cso_log = CrowSearchAlgorithm(problem_dim=1, population_size=20, awareness_prob=0.1, flight_length=2,lower_bound=[0.0001], upper_bound=[1.0], max_iter=50)
par = cso_log.optimize(fitness_log,data,X_train,y_train, X_test, y_test)
log_reg = LogisticRegression(C= par[0])
log_reg.fit(X_train,y_train)
prediction = log_reg.predict(X_test)
pre = log_reg.predict(X_train)
cm = confusion_matrix(prediction, y_test)
print("预测准确率为：",np.sum(prediction==y_test)/len(y_test))
acc['log'] = np.sum(prediction==y_test)/len(y_test)
stacking.append(prediction)

#

#KNN
cso_KNN = CrowSearchAlgorithm(problem_dim=1, population_size=20, awareness_prob=0.1, flight_length=2,lower_bound=[1], upper_bound=[100], max_iter=50)
par = cso_KNN.optimize(fitness_KNN,data,X_train,y_train, X_test, y_test)
knn = KNNClassify(k = int(par[0])) # k = 20
knn.fit(X_train,y_train)
prediction = knn.predict(X_test)
pre = knn.predict(X_train)
cm = confusion_matrix(prediction, y_test)
print("预测准确率为：",np.sum(prediction==y_test)/len(y_test))
acc['KNN'] = np.sum(prediction==y_test)/len(y_test)
stacking.append(prediction)


#SVM
'''
svm = SVM(kernel='rbf', k=1)
svm.fit(X_train, y_train, eval_train=True)
prediction =svm.predict(X_test)[0].astype(np.int64)
prediction[prediction == -1] = 0
pre = svm.predict(X_train)[0]
pre[pre == -1] = 0
y_train[y_train == -1] =0
stacking.append(prediction)
'''
#XGBoost
cso_XGBoost = CrowSearchAlgorithm(problem_dim=3, population_size=10, awareness_prob=0.1, flight_length=2,lower_bound=[50,3,0], upper_bound=[500,10,5], max_iter=20)
par = cso_XGBoost.optimize(fitness_XGboost,data,X_train,y_train, X_test, y_test)

model = XGBClassifier(n_estimators = int(par[0]),max_depth = int(par[1]), gamma =par[2])
model.fit(X_train, y_train)
prediction = model.predict(X_test)

cm = confusion_matrix(prediction, y_test)
print("预测准确率为：",np.sum(prediction==y_test)/len(y_test))
acc['XGBoost'] = np.sum(prediction==y_test)/len(y_test)
stacking.append(prediction)


#smo -svm
cso_svm = CrowSearchAlgorithm(problem_dim=2, population_size=20, awareness_prob=0.1, flight_length=2,lower_bound=[0.001,0.0001], upper_bound=[10,1], max_iter=20)
par = cso_svm.optimize(fitness_svm,data,X_train,y_train, X_test, y_test)

model = svm_train(y_train, X_train, f'-c {par[0]} -g {par[1]}')
prediction, p_acc, p_val = svm_predict(y_test, X_test, model)
ACC, MSE, SCC = p_acc
cm = confusion_matrix(prediction, y_test)
print("预测准确率为：",np.sum(prediction==y_test)/len(y_test))
acc['SVM'] = np.sum(prediction==y_test)/len(y_test)

stacking.append(prediction)

#mlp
cso_mlp = CrowSearchAlgorithm(problem_dim=5, population_size=4, awareness_prob=0.1, flight_length=2,lower_bound=[0.1,0.1,0.1,16,0.0001], upper_bound=[0.5,0.5,0.5,512,0.1], max_iter=4)
par = cso_mlp.optimize(fitness_mlp,data,X_train,y_train, X_test, y_test)

mlp = nn.Sequential(nn.Flatten(),
                    nn.Linear(X_train.shape[1], 64),
                    nn.ReLU(),
                    nn.Dropout(p=par[0]),
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Dropout(p=par[1]),
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Dropout(p=par[2]),
                    nn.Linear(64, len(np.unique(y_train)))
                    )

prediction = mlp_train(mlp , batch_size = int(par[3]) , num_epochs = 1000 , lr = par[4] , X = X_train, y = y_train, X_test= X_test, y_test= y_test)
cm = confusion_matrix(prediction, y_test)
print("预测准确率为：",np.sum(prediction==y_test)/len(y_test))
acc['mlp'] = np.sum(prediction==y_test)/len(y_test)

stacking.append(prediction)




#检查过拟合
cm_train = confusion_matrix(pre, y_train)
print(cm_train)




#stacking
stacking.append(y_test)
stacking = list(map(list, zip(*stacking)))

#将投票作为次级学习器
prediction = []
for row in stacking:
    count = np.bincount(row[:-1])
    prediction.append(count.argmax())

cm = confusion_matrix(prediction, y_test)
print("预测准确率为：",np.sum(prediction==y_test)/len(y_test))
acc['stacking'] = np.sum(prediction==y_test)/len(y_test)

stacking =  pd.DataFrame(stacking)
train, test = train_test_split(stacking, test_size=0.4)
X_train = np.array(train.drop(stacking.columns[-1],axis=1)).astype(float)
y_train = np.array(train[stacking.columns[-1]])
X_test = np.array(test.drop(stacking.columns[-1],axis=1)).astype(float)
y_test = np.array(test[stacking.columns[-1]])



mlp = nn.Sequential(nn.Flatten(),
                    nn.Linear(X_train.shape[1], 8),
                    nn.ReLU(),
                    nn.Dropout(p=0.1),
                    nn.Linear(8, len(np.unique(y_train)))
                    )

prediction = mlp_train(mlp , batch_size = 8 , num_epochs = 500 , lr = 0.01 , X = X_train, y = y_train, X_test= X_test, y_test= y_test)
acc['st_mlp'] = np.sum(prediction==y_test)/len(y_test)

cart = DecisionTree(max_depth= 20,m = 1)
cart.fit(X_train,y_train)
prediction = cart.predict(X_test)
pre = cart.predict(X_train)
cm = confusion_matrix(prediction, y_test)
print("预测准确率为：",np.sum(prediction==y_test)/len(y_test))
acc['st_cart'] = np.sum(prediction==y_test)/len(y_test)

'''
import csv
filename = 'stacking_data_0511.csv'
with open(filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    for i in stacking:
        csvwriter.writerow(i)



# 记录准确率
import csv
with open(f'{data}acc.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    writer.writerow(["model", "acc"])
    for key, value in acc.items():
        writer.writerow([key, value])  
'''
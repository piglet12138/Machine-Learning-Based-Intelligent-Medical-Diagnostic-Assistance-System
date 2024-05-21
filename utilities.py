import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



class Data_Set:
    def __init__(self):

        heart_disease = pd.read_csv("./_internal/heart_disease.csv")
        heart_disease = heart_disease.dropna()
        heart_disease = heart_disease[['age','chol','oldpeak','thalach','ca','thal','trestbps','num']]

        breast_cancer = pd.read_csv("./_internal/breast_cancer.csv")
        breast_cancer[breast_cancer == 'M'] = 0
        breast_cancer[breast_cancer == 'B'] = 1
        breast_cancer = breast_cancer.dropna()
        breast_cancer = breast_cancer[['area3','concave_points1','concave_points3','concavity3','perimeter3','radius3',
        'compactness1','smoothness2','concavity2','concave_points2','fractal_dimension3','Diagnosis']]

        dermatology = pd.read_csv("./_internal/dermatology.csv")
        dermatology = dermatology.dropna()
        dermatology = dermatology[['scaling','itching','koebner phenomenon','follicular papules','eosinophils in the infiltrate',
        'pnl infiltrate','fibrosis of the papillary dermis','clubbing of the rete ridges','thinning of the suprapapillary epidermis',
        'munro microabcess','elongation of the rete ridges','saw-tooth appearance of retes','spongiosis','class']]
        dermatology['class'] -= 1

        liver = pd.read_csv("./_internal/indian_liver_patient.csv")
        liver[liver == 'Male'] = 0.0
        liver[liver == 'Female'] = 1.0
        liver = liver.dropna()

        ada_liver = liver[
            ['Age', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Albumin',
             'Dataset']]

        diabetes = pd.read_csv("./_internal/diabetes.csv")
        diabetes = diabetes.dropna()
        diabetes = diabetes[['Pregnancies','Glucose','BloodPressure','BMI','DiabetesPedigreeFunction','Outcome']]

        diabetes_2 = pd.read_csv("./_internal/diabetes_prediction_dataset.csv")
        diabetes_2[diabetes_2 == 'Male'] = 0.0
        diabetes_2[diabetes_2 == 'Female'] = 1.0
        diabetes_2[diabetes_2 == 'Other'] = 2.0

        diabetes_2[diabetes_2 == 'No Info'] = 0.0
        diabetes_2[diabetes_2 == 'never'] = 0.0
        diabetes_2[diabetes_2 == 'former'] = 1.0
        diabetes_2[diabetes_2 == 'not current'] = 1.0
        diabetes_2[diabetes_2 == 'current'] = 2.0
        diabetes_2[diabetes_2 == 'ever'] = 3.0
        diabetes_2 = diabetes_2.dropna()

        self.dataset = {'heart_disease':heart_disease,'breast_cancer':breast_cancer,'dermatology':dermatology,
                   'liver_disease':ada_liver, 'diabetes1':diabetes,'diabetes2':diabetes_2}
    def load_data(self,target_data):
        data = self.dataset[target_data]

        train, test = train_test_split(data, test_size=0.2)

        X_train = np.array(train.drop(data.columns[-1], axis=1)).astype(float)
        y_train = np.array(train[data.columns[-1]]).astype(int)

        '''
        smote = SMOTE(sampling_strategy='auto')  # 'auto'表示过采样少数类别与多数类别数目相等
        X_train, y_train = smote.fit_resample(X_train , y_train)
        '''

        X_test = np.array(test.drop(data.columns[-1], axis=1)).astype(float)
        y_test = np.array(test[data.columns[-1]]).astype(int)

        X_mean = X_train.mean(axis=0)
        X_std = X_train.std(axis=0)
        X_train = (X_train - X_mean) / X_std
        X_test = (X_test - X_mean) / X_std
        return X_train,y_train, X_test, y_test


# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 09:14:10 2020

@author: Souridas A
"""

import numpy as np
import pandas as pd

# Importing the dataset
test = pd.read_csv('test_2umaH9m.csv')
train=pd.read_csv('train_LZdllcl.csv')
from sklearn.preprocessing import Imputer
X_train = train.iloc[:, 1:13].values
X_test = test.iloc[:, 1:13].values
y_train = train.iloc[:, -1].values
imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent',axis=0)
imputer = imputer.fit(X_train[:, 7:8])
X_train[:, 7:8] = imputer.transform(X_train[:, 7:8])
imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent',axis=0)
imputer = imputer.fit(X_test[:, 7:8])
X_test[:, 7:8] = imputer.transform(X_test[:, 7:8])
from sklearn_pandas import CategoricalImputer
imputer =CategoricalImputer()
imputer = imputer.fit(X_train[:, 2])
X_train[:, 2] = imputer.transform(X_train[:, 2])
imputer = CategoricalImputer()
imputer = imputer.fit(X_test[:, 2])
X_test[:, 2] = imputer.transform(X_test[:, 2])
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X_train[:, 0] = labelencoder_X_1.fit_transform(X_train[:, 0])
labelencoder_X_2 = LabelEncoder()
X_train[:, 1] = labelencoder_X_2.fit_transform(X_train[:, 1])
labelencoder_X_3 = LabelEncoder()
X_train[:, 2] = labelencoder_X_3.fit_transform(X_train[:, 2])
labelencoder_X_4 = LabelEncoder()
X_train[:, 3] = labelencoder_X_4.fit_transform(X_train[:, 3])
labelencoder_X_5 = LabelEncoder()
X_train[:, 4] = labelencoder_X_5.fit_transform(X_train[:, 4])
labelencoder_X_6 = LabelEncoder()
X_test[:, 0] = labelencoder_X_6.fit_transform(X_test[:, 0])
labelencoder_X_7 = LabelEncoder()
X_test[:, 1] = labelencoder_X_7.fit_transform(X_test[:, 1])
labelencoder_X_8 = LabelEncoder()
X_test[:, 2] = labelencoder_X_8.fit_transform(X_test[:, 2])
labelencoder_X_9 = LabelEncoder()
X_test[:, 3] = labelencoder_X_9.fit_transform(X_test[:, 3])
labelencoder_X_10 = LabelEncoder()
X_test[:, 4] = labelencoder_X_10.fit_transform(X_test[:, 4])
onehotencoder=OneHotEncoder(categorical_features =[0])
X_train=onehotencoder.fit_transform(X_train).toarray()
X_train=X_train[:,1:]
onehotencoder1=OneHotEncoder(categorical_features =[8])
X_train=onehotencoder1.fit_transform(X_train).toarray()
X_train=X_train[:,1:]
onehotencoder2=OneHotEncoder(categorical_features =[41])
X_train=onehotencoder2.fit_transform(X_train).toarray()
X_train=X_train[:,1:]
onehotencoder3=OneHotEncoder(categorical_features =[44])
X_train=onehotencoder3.fit_transform(X_train).toarray()
X_train=X_train[:,1:]
onehotencoder=OneHotEncoder(categorical_features =[0])
X_test=onehotencoder.fit_transform(X_test).toarray()
X_test=X_test[:,1:]
onehotencoder1=OneHotEncoder(categorical_features =[8])
X_test=onehotencoder1.fit_transform(X_test).toarray()
X_test=X_test[:,1:]
onehotencoder2=OneHotEncoder(categorical_features =[41])
X_test=onehotencoder2.fit_transform(X_test).toarray()
X_test=X_test[:,1:]
onehotencoder3=OneHotEncoder(categorical_features =[44])
X_test=onehotencoder3.fit_transform(X_test).toarray()
X_test=X_test[:,1:]
m=X_train[:,48]+X_train[:,50]+X_train[:,51]+X_train[:,52]
m=m.reshape((54808,1))
c=np.append(X_train,m,axis=1)
r=X_train[:,46]*X_train[:,52]
r=r.reshape((54808,1))
c=np.append(c,r,axis=1)
X_train=np.delete(c,45,1)
a=X_test[:,48]+X_test[:,50]+X_test[:,51]+X_test[:,52]
a=a.reshape((23490,1))
b=np.append(X_test,a,axis=1)
r=X_test[:,46]*X_test[:,52]
r=r.reshape((23490,1))
b=np.append(b,r,axis=1)
X_test=np.delete(b,45,1)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#Finding number of principal components
from sklearn.decomposition import PCA
pca=PCA().fit(X_train)
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"]=(12,6)
fig,ax=plt.subplots()
xi=np.arange(1,55,step=1)
y=np.cumsum(pca.explained_variance_ratio_)
plt.ylim(0.0,1.1)
plt.plot(xi,y,marker='o',linestyle='--',color='b')
plt.xlabel('Number of Components')
plt.xticks(np.arange(0,55,step=1))
plt.ylabel("Cumulative variance(%)")
plt.title('Number of components needed to explain variance')
plt.axhline(y=.95,color='r',linestyle='-')
plt.text(0.5,0.85,'95% cut-off threshold',color='red',fontsize=16)
ax.grid(axis='x')
plt.show()
#Fit transforrm dataset using pca
pca=PCA(n_components=35)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
#Training data using ANN

import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 18, init = 'uniform', activation = 'relu', input_dim = 35))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 18, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 18, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 18, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size =8, nb_epoch =50)

# Part 3 - Making the predictions and evaluating the model
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
y_pred=y_pred*1

'''from sklearn.model_selection import GridSearchCV
parameters = [{'batch_size':[5,15,10],'nb_epoch':[10,100,150]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 3,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_'''

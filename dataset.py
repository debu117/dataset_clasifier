#checking the versions of various libraries and python
import sys #to check versions
print(' Python: {}'.format(sys.version))
import scipy
print(' Scipy: {}'.format(sys.version))
import numpy
print(' Numpy: {}'.format(sys.version))
import matplotlib
print(' Matplotlib: {}'.format(sys.version))
import pandas
print(' Pandas: {}'.format(sys.version))
import sklearn
print(' Sklearn: {}'.format(sys.version))
 Python: 3.7.11 (default, Jul  3 2021, 18:01:19) 
[GCC 7.5.0]
 Scipy: 3.7.11 (default, Jul  3 2021, 18:01:19) 
[GCC 7.5.0]
 Numpy: 3.7.11 (default, Jul  3 2021, 18:01:19) 
[GCC 7.5.0]
 Matplotlib: 3.7.11 (default, Jul  3 2021, 18:01:19) 
[GCC 7.5.0]
 Pandas: 3.7.11 (default, Jul  3 2021, 18:01:19) 
[GCC 7.5.0]
 Sklearn: 3.7.11 (default, Jul  3 2021, 18:01:19) 
[GCC 7.5.0]
#importing libraries
from pandas.plotting import scatter_matrix
import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier
#loading the datasaet
iris_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
names=['sepal_length','sepal_width','petal_length','petal_width','class']
data= pd.read_csv(iris_url,names=names)
#getting to know the data
print('Dimension: ', data.shape)
print("first few rows:")
print(data.head(15))
Dimension:  (150, 5)
first few rows:
    sepal_length  sepal_width  petal_length  petal_width        class
0            5.1          3.5           1.4          0.2  Iris-setosa
1            4.9          3.0           1.4          0.2  Iris-setosa
2            4.7          3.2           1.3          0.2  Iris-setosa
3            4.6          3.1           1.5          0.2  Iris-setosa
4            5.0          3.6           1.4          0.2  Iris-setosa
5            5.4          3.9           1.7          0.4  Iris-setosa
6            4.6          3.4           1.4          0.3  Iris-setosa
7            5.0          3.4           1.5          0.2  Iris-setosa
8            4.4          2.9           1.4          0.2  Iris-setosa
9            4.9          3.1           1.5          0.1  Iris-setosa
10           5.4          3.7           1.5          0.2  Iris-setosa
11           4.8          3.4           1.6          0.2  Iris-setosa
12           4.8          3.0           1.4          0.1  Iris-setosa
13           4.3          3.0           1.1          0.1  Iris-setosa
14           5.8          4.0           1.2          0.2  Iris-setosa
#brief statistical description
data.describe()
sepal_length	sepal_width	petal_length	petal_width
count	150.000000	150.000000	150.000000	150.000000
mean	5.843333	3.054000	3.758667	1.198667
std	0.828066	0.433594	1.764420	0.763161
min	4.300000	2.000000	1.000000	0.100000
25%	5.100000	2.800000	1.600000	0.300000
50%	5.800000	3.000000	4.350000	1.300000
75%	6.400000	3.300000	5.100000	1.800000
max	7.900000	4.400000	6.900000	2.500000
#class distribution
data.groupby('class').size()
class
Iris-setosa        50
Iris-versicolor    50
Iris-virginica     50
dtype: int64
Time for data visualization

#univariate plots - box and whisker plots
data.plot(kind='box', subplots=True , layout=(2,2), sharex=False, sharey=False)
pyplot.show()

#univariate plot - histogram
data.hist()
pyplot.show()
#multivariate plots
scatter_matrix(data)
pyplot.show()

#creating a valid dataset
#splitting the dataset into test and train
arr=data.values
X=arr[:,0:4]
y= arr[:,4]
X_train,X_validation,Y_train,Y_validation=train_test_split(X,y,test_size=0.2,random_state=1)
#will be trying out the following algorithm and choose the best amongst it
#logistic regression
#Linear Discriminant Analysis
#K-Nearest neighbours
#classification and Regression Tree
#gaussian Naive Bayes
#Support Vector Machine
#Building models
models=[]
models.append(('LR',LogisticRegression(solver='liblinear',multi_class='ovr')))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC(gamma='auto')))
#Evaluate the models
results=[]
names=[]
for name,model in models:
  kfold=StratifiedKFold(n_splits=10) #was showing some warning when using random state
  cv_results= cross_val_score(model, X_train, Y_train,cv=kfold, scoring='accuracy')
  results.append(cv_results)
  names.append(name)
  print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
LR: 0.950000 (0.055277)
LDA: 0.975000 (0.038188)
KNN: 0.958333 (0.041667)
NB: 0.950000 (0.055277)
SVM: 0.983333 (0.033333)
#compare the models
pyplot.boxplot(results,labels=names)
pyplot.title('Algorithms Comparison')
pyplot.show()

#making predictions on SVM as it has the highest accuracy when training
model=SVC(gamma='auto')
model.fit(X_train,Y_train)
predictions= model.predict(X_validation)
#evaluating the predictions
print(accuracy_score(Y_validation,predictions))
print(confusion_matrix(Y_validation,predictions))
print(classification_report(Y_validation,predictions))
0.9666666666666667
[[11  0  0]
 [ 0 12  1]
 [ 0  0  6]]
                 precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        11
Iris-versicolor       1.00      0.92      0.96        13
 Iris-virginica       0.86      1.00      0.92         6

       accuracy                           0.97        30
      macro avg       0.95      0.97      0.96        30
   weighted avg       0.97      0.97      0.97        30

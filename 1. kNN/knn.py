import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import os
os.sys.path.append('..')



## Preprocess
df = pd.read_csv("1. kNN\dataset\winequalityN.csv")
df.shape
df.head()

df.isnull().sum()
df.dropna(inplace=True)
df.reset_index(drop=True)

df['quality'].value_counts()

Y=df['quality'].apply(lambda x: 1 if x > 5 else 0 )
X=df.drop(['quality'], axis=1)
X['type'] = X['type'].apply(lambda x: 1 if x == 'white' else 10000)



## Build and train model
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.2,random_state=42)

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

Y = []
for i in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=i)
    KNN=knn.fit(X_train,Y_train)
    prediction=KNN.predict(X_test)
    accuracy = metrics.accuracy_score(Y_test,prediction)
    print("K={} accuracy: {}".format(i, accuracy))
    Y.append(1 - accuracy)



# Show kcount-error plot
X = range(1, 20)
plt.scatter(X, Y)
plt.plot(X, Y)
plt.xlabel("K count")
plt.ylabel("Error")
plt.grid()
plt.xticks([0,5,10,15,20])
plt.show()


# When k=1, show result and accuracy
knn = KNeighborsClassifier(n_neighbors=1)
KNN=knn.fit(X_train,Y_train)
prediction=KNN.predict(X_test)
accuracy = metrics.accuracy_score(Y_test,prediction)
print(prediction)
print("Accuracy:",accuracy)
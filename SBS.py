from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

class SBS():
    """Clase SBS"""
    def __init__(self, estimator,k_features,
                 scoring=accuracy_score, test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state  =random_state

    def fit(self,X,y):
        """ Fit SBS"""
        X_train,X_test,y_train,y_test=\
            train_test_split(X,y,test_size = self.test_size,random_state = self.random_state)
        
        dim =X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score =self._calc_score(X_train,y_train,X_test,y_test,self.indices_)

        self.scores_ = [score]
        
        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_,r=dim -1 ):
                score = self._calc_score(X_train,y_train,X_test,y_test,p)
                scores.append(score)
                subsets.append(p)
            
            best = np.argmax(scores)
            self.indices_= subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]
        return self

    def transform(self,X):
        """ Transform"""
        return X[:,self.indices_]

    def _calc_score(self,X_train,y_train,X_test,y_test, indices):
        """Score"""
        self.estimator.fit(X_train[:, indices],y_train)
        y_pred = self.estimator.predict(X_test[:,indices])
        score = self.scoring(y_test,y_pred)
        return score
    

"""KNN con SBS"""

wine=load_wine()
X=wine.data
y=wine.target
columns=wine.feature_names
print(columns)

sc=StandardScaler()
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,test_size=0.3,stratify=y)
X_train_std=sc.fit_transform(X_train)
X_test_std=sc.transform(X_test)


knn=KNeighborsClassifier(n_neighbors=5)
sbs=SBS(knn,k_features=1)
sbs.fit(X_train_std,y_train)

K_feat=[len(k) for k in sbs.subsets_]
plt.plot(K_feat,sbs.scores_,marker='o')
plt.ylim([0.7,1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of Features')
plt.grid()
plt.show()

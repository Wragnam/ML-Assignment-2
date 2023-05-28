from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import pandas as pd




def testing_data():
    # testing_data = pd.read_csv(str(file), sep=",",header=None,index_col=None)
    training_data = pd.read_csv('traindata.txt',sep=",",header=None, index_col=None)

    train_labels = pd.read_csv('trainlabels.txt',sep=",",index_col=False,header = None)

    logistic_regression = LogisticRegression()
    logistic_regression.max_iter = 10000
    logistic_regression.fit(training_data,train_labels.values.ravel())

    linear_discriminator = LinearDiscriminantAnalysis()
    linear_discriminator.max_iter = 10000
    linear_discriminator.fit(training_data,train_labels)
    


    

    out = logistic_regression.predict(training_data)

    return out

df = pd.DataFrame(testing_data())

df.to_csv("out.csv",index=False,header=None)





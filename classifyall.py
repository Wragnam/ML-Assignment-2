from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import pandas as pd
import sklearn




def testing_data():
    # testing_data = pd.read_csv(str(file), sep=",",header=None,index_col=None)
    training_data = pd.read_csv('traindata.txt',sep=",",header=None, index_col=None)

    train_labels = pd.read_csv('trainlabels.txt',sep=",",index_col=False,header = None)

    #Logistic Regression Method
    logistic_regression = LogisticRegression()
    logistic_regression.max_iter = 10000
    logistic_regression.fit(training_data,train_labels.values.ravel())
    

    #Linear Desciminator Method
    linear_discriminator = LinearDiscriminantAnalysis()
    linear_discriminator.max_iter = 10000
    linear_discriminator.fit(training_data,train_labels)
    


    
    #Outputs

    #Linear Regression
    out_logistic_regression = logistic_regression.predict(training_data)
    logistic_regression_accuracy = sklearn.metrics.accuracy_score(train_labels.values.ravel(),out_logistic_regression)

    #Linear Discriminator
    out_linear_discriminator = linear_discriminator.predict(training_data)
    linear_discriminator_accuracy = sklearn.metrics.accuracy_score(train_labels.values.ravel(),out_linear_discriminator)


    return linear_discriminator_accuracy


print(testing_data())


####Logistic regression accuracy = 0.2112
####Linear discriminator accuracy = 0.2145


#Return output
# df = pd.DataFrame(testing_data())

# df.to_csv("out.csv",index=False,header=None)





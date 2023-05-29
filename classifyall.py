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

    #KNeighborsClassifier Method
    KNeighbours_method = KNeighborsClassifier()
    KNeighbours_method.max_iter = 10000
    KNeighbours_method.fit(training_data,train_labels.values.ravel())
    

    #Linear Desciminator Method
    linear_discriminator = LinearDiscriminantAnalysis()
    linear_discriminator.max_iter = 10000
    linear_discriminator.fit(training_data,train_labels.values.ravel())

    #Decision Tree Classifier
    desicion_tree_classifier = DecisionTreeClassifier()
    desicion_tree_classifier.max_iter = 10000
    desicion_tree_classifier.fit(training_data, train_labels.values.ravel())

    #SVM prediction
    svc = SVC()
    svc.max_iter = 10000
    svc.fit(training_data, train_labels.values.ravel())
    
    #GaussianNB Classifier
    gaussian_classifier = GaussianNB()
    gaussian_classifier.max_iter =10000
    gaussian_classifier.fit(training_data,train_labels.values.ravel())

    
    #Outputs

    #Linear Regression
    out_logistic_regression = logistic_regression.predict(training_data)
    logistic_regression_accuracy = sklearn.metrics.accuracy_score(train_labels.values.ravel(),out_logistic_regression)

    #Linear Discriminator
    out_linear_discriminator = linear_discriminator.predict(training_data)
    linear_discriminator_accuracy = sklearn.metrics.accuracy_score(train_labels.values.ravel(),out_linear_discriminator)

    #KNeighborsClassifier
    out_KNeighbors = KNeighbours_method.predict(training_data)
    KNeighbours_accuracy = sklearn.metrics.accuracy_score(train_labels.values.ravel(),out_KNeighbors)

    #Decision Tree
    out_Desicion = desicion_tree_classifier.predict(training_data)
    decision_tree_accuracy = sklearn.metrics.accuracy_score(train_labels.values.ravel(), out_Desicion)

    #SVC
    out_svc = svc.predict(training_data)
    svc_accuracy = sklearn.metrics.accuracy_score(train_labels.values.ravel(), out_svc)

    #Guasian
    out_guasian = gaussian_classifier.predict(training_data)
    guassian_accuracy_score =sklearn.metrics.accuracy_score(train_labels.values.ravel(), out_guasian)

    return guassian_accuracy_score


print(testing_data())


####Logistic regression accuracy = 0.2112
####Linear discriminator accuracy = 0.2145
####KNeighbors accuracy = 0.4403
####Decitional accuracy = 1.0
####SVC accuracy = 0.3751


#Return output
# df = pd.DataFrame(testing_data())

# df.to_csv("out.csv",index=False,header=None)





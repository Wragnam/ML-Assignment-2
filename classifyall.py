from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import pandas as pd
import sklearn
import time




def testing_data(testing_data):
    # testing_data = pd.read_csv(str(file), sep=",",header=None,index_col=None)
    training_data = pd.read_csv('traindata.txt',sep=",",header=None, index_col=None)

    train_labels = pd.read_csv('trainlabels.txt',sep=",",index_col=False,header = None)

    # start = time.time()
    # #Logistic Regression Method
    # logistic_regression = LogisticRegression()
    # logistic_regression.max_iter = 10000
    # logistic_regression.fit(training_data,train_labels.values.ravel())
    # #Logistic Regression
    # out_logistic_regression = logistic_regression.predict(training_data)
    # logistic_regression_accuracy = sklearn.metrics.accuracy_score(train_labels.values.ravel(),out_logistic_regression)
    # end = time.time()
    # logistic_regression_time = end-start

    # start = time.time()
    # #KNeighborsClassifier Method
    # KNeighbours_method = KNeighborsClassifier()
    # KNeighbours_method.max_iter = 10000
    # KNeighbours_method.fit(training_data,train_labels.values.ravel())
    #  #KNeighborsClassifier
    # out_KNeighbors = KNeighbours_method.predict(training_data)
    # KNeighbours_accuracy = sklearn.metrics.accuracy_score(train_labels.values.ravel(),out_KNeighbors)
    # end = time.time()
    # KNeighbours_time = end - start


    # start = time.time()
    # #Linear Desciminator Method
    # linear_discriminator = LinearDiscriminantAnalysis()
    # linear_discriminator.max_iter = 10000
    # linear_discriminator.fit(training_data,train_labels.values.ravel())
    #  #Linear Discriminator
    # out_linear_discriminator = linear_discriminator.predict(training_data)
    # linear_discriminator_accuracy = sklearn.metrics.accuracy_score(train_labels.values.ravel(),out_linear_discriminator)
    # end = time.time()
    # linear_discriminator_time = end-start

    #Decision Tree Classifier
    desicion_tree_classifier = DecisionTreeClassifier()
    desicion_tree_classifier.max_iter = 10000
    desicion_tree_classifier.fit(training_data, train_labels.values.ravel())
    #Decision Tree
    out_Desicion = desicion_tree_classifier.predict(testing_data)
    decision_tree_accuracy = sklearn.metrics.accuracy_score(train_labels.values.ravel(), out_Desicion)
    

    # start = time.time()
    # #SVM prediction
    # svc = SVC()
    # svc.max_iter = 10000
    # svc.fit(training_data, train_labels.values.ravel())
    # #SVC
    # out_svc = svc.predict(training_data)
    # svc_accuracy = sklearn.metrics.accuracy_score(train_labels.values.ravel(), out_svc)
    # end = time.time()
    # svc_time = end-start
    

    # start = time.time()
    # #GaussianNB Classifier
    # gaussian_classifier = GaussianNB()
    # gaussian_classifier.max_iter =10000
    # gaussian_classifier.fit(training_data,train_labels.values.ravel())
    #  #Guasian
    # out_guasian = gaussian_classifier.predict(training_data)
    # guassian_accuracy_score =sklearn.metrics.accuracy_score(train_labels.values.ravel(), out_guasian)
    # end = time.time()
    # guassian_time = end-start

    return 0


print(testing_data())


####Logistic regression accuracy = 0.2112, 1.23s
####Linear discriminator accuracy = 0.2145, 0.15s
####KNeighbors accuracy = 0.4403, 0.78s
####Decisional accuracy = 1.0, 0.75s
####SVC accuracy = 0.3751, 26.00s
####Guassian accuracy = 0.2239, 0.23s


#Return output
# df = pd.DataFrame(testing_data())

# df.to_csv("out.csv",index=False,header=None)





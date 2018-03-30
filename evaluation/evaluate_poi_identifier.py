#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
from time import time

from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier()

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time() - t0, 3), "s"
time1 = time()
pred=clf.predict(features_test)
print "testing time:", round(time()-time1, 3), "s"
accuracy=accuracy_score(labels_test,pred)
print accuracy

print ("Number of POI's: ",len([i for i in labels_test if int(i)==1]))
print ("Number of people in test sest: ",len(labels_test))
print ("Actual -> Predicted")
print ("Precision: ",precision_score([int(i) for i in pred],[int(i) for i in labels_test]))
print ("Recall: ",recall_score([int(i) for i in pred],[int(i) for i in labels_test]))





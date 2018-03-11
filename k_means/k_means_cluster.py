#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""




import pickle
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit




def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()



### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)
Min = 10000000
Max = 0
for k in data_dict:
    stock = data_dict[k]["exercised_stock_options"]
    if stock != 'NaN':
        if stock < Min:
            Min = stock
        if stock > Max:
            Max = stock

print "min:", Min
print "max:", Max

Min1 = 10000000
Max1 = 0
for k in data_dict:
    sal = data_dict[k]["salary"]
    if sal != 'NaN':
        if sal < Min1:
            Min1 = sal
        if sal > Max1:
            Max1 = sal

print "min:", Min1
print "max:", Max1


### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
#feature_3 = "total_payments"
poi  = "poi"
features_list = [poi, feature_1, feature_2]
#features_list = [poi, feature_1, feature_2,feature_3]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data = featureFormat(data_dict, features_list)
data_scaled = scaler.fit_transform(data)
poi, finance_features = targetFeatureSplit(data_scaled )

# predicting scalar works
# predicting scalar works
scaler1 = MinMaxScaler()
data3 = featureFormat(data_dict, [feature_1, feature_2])
data3_scaled = scaler1.fit_transform(data3)
print scaler1.transform([[200000.,1000000.]])

### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)
#for f1, f2,_ in finance_features:
#for f1, f2,_ in finance_features:
for f1, f2 in finance_features:
    plt.scatter(f1, f2)
plt.show()

### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred
from sklearn.cluster import KMeans
clf=KMeans(n_clusters=2)
pred = clf.fit_predict( finance_features )

### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print "no predictions object named pred found, no clusters to plot"

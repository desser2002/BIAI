

import sys
import pickle
sys.path.append("../tools/")
import matplotlib.pyplot as plt
import numpy as np
import random


from tester import dump_classifier_and_data

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif



def featureFormat(dictionary, features, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False,
                  sort_keys=False):
    

    return_list = []

    # Key order - first branch is for Python 3 compatibility on mini-projects,
    # second branch is for compatibility on final project.
    if isinstance(sort_keys, str):
        
        keys = pickle.load(open(sort_keys, "rb"))
    elif sort_keys:
        keys = sorted(dictionary.keys())
    else:
        keys = dictionary.keys()

    for key in keys:
        tmp_list = []
        for feature in features:
            try:
                dictionary[key][feature]
            except KeyError:
                print("error: key ", feature, " not present")
                return
            value = dictionary[key][feature]
            if value == "NaN" and remove_NaN:
                value = 0
            tmp_list.append(float(value))

       
        append = True
       
        if features[0] == 'poi':
            test_list = tmp_list[1:]
        else:
            test_list = tmp_list
        
        if remove_all_zeroes:
            append = False
            for item in test_list:
                if item != 0 and item != "NaN":
                    append = True
                    break
        
        if remove_any_zeroes:
            if 0 in test_list or "NaN" in test_list:
                append = False
        
        if append:
            return_list.append(np.array(tmp_list))

    return np.array(return_list)


def targetFeatureSplit(data):
    
    target = []
    features = []
    for item in data:
        target.append(item[0])
        features.append(item[1:])

    return target, features

features_list = ['poi',
                 'salary',
                 'bonus',
                 'deferral_payments',
                 'expenses',
                 'deferred_income',
                 'long_term_incentive',
                 'restricted_stock_deferred',
                 'shared_receipt_with_poi',
                 'loan_advances',
                 'from_messages',
                 'director_fees',
                 'total_stock_value',
                 'from_poi_to_this_person',
                 'from_this_person_to_poi',
                 'total_payments',
                 'exercised_stock_options',
                 'to_messages',
                 'restricted_stock',
                 'other'] 


### Load the dictionary containing the dataset
with open("my_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)


print ('Total number of ppl:', len(data_dict) )#How many people in the dataset
print ('Number of features:', len(features_list)) #How mant features in the list

poi_num = 0
for i in data_dict:
    if data_dict[i]['poi'] == True:
        poi_num += 1
print ('Number of poi:', poi_num) #Number of poi


### Create new salary-bonus-ratio, from_this_person_to_poi % & from_poi_to_this_person % features

def ratio_calc(numerator, denominator):
    fraction = 0
    if numerator == 'NaN' or denominator == 'NaN':
        fraction == 'NaN'
    else:
        fraction = float(numerator) / float(denominator)
    return fraction

for name in data_dict:
    salary_bonus_ratio_temp = ratio_calc(data_dict[name]['salary'], data_dict[name]['bonus'])
    data_dict[name]['salary_bonus_ratio'] = salary_bonus_ratio_temp

    from_this_person_to_poi_ratio_temp = ratio_calc(data_dict[name]['from_this_person_to_poi'], data_dict[name]['from_messages'])
    data_dict[name]['from_this_person_to_poi_ratio'] = from_this_person_to_poi_ratio_temp

    from_poi_to_this_person_ratio_temp = ratio_calc(data_dict[name]['from_poi_to_this_person'], data_dict[name]['to_messages'])
    data_dict[name]['from_poi_to_this_person_ratio'] = from_poi_to_this_person_ratio_temp


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
features_list = ['poi',
                 'salary',
                 'bonus',
                 #'deferral_payments',
                 #'expenses',
                 'deferred_income',
                 'long_term_incentive',
                 #'restricted_stock_deferred',
                 'shared_receipt_with_poi',
                 #'loan_advances',
                 #'from_messages',
                 #'director_fees',
                 'total_stock_value',
                 #'from_poi_to_this_person',
                 #'from_this_person_to_poi',
                 'total_payments',
                 'exercised_stock_options',
                 #'from_poi_to_this_person_ratio',
                 'from_this_person_to_poi_ratio',
                 #'salary_bonus_ratio',
                 #'to_messages',
                 'restricted_stock'
                 #'other'
                 ] 
### Other is removed from the dataset as it doesn't tell much and including it
### might skew the predictions

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Applying feature scaling using MinMaxScaler

scaler = MinMaxScaler()
features = scaler.fit_transform(features)


### Using SelectKBest to determine by features to use
'''
selector = SelectKBest(f_classif, k=20)
selector.fit(features, labels)
features = selector.transform(features)
feature_scores = zip(features_list[1:],selector.scores_)

sorted_scores = sorted(feature_scores, key=lambda feature: feature[1], reverse = True)
for item in sorted_scores:
    print item[0], item[1]
'''

### Using Decision Tree
'''
clf = DecisionTreeClassifier()
clf.fit(features, labels)
dt_scores = zip(features_list[1:],clf.feature_importances_)
sorted_dtscores = sorted(dt_scores, key=lambda feature: feature[1], reverse = True)
for item in sorted_dtscores:
    print item[0], item[1]
'''


### Naive-Bayes Gaussian
'''
clf = GaussianNB()
'''

### Decision Tree
#clf = tree.DecisionTreeClassifier()

### AdaBoost Classifier
#clf = AdaBoostClassifier()

### K Nearest Neighbors Classifier
#clf = KNeighborsClassifier()

### Random Forest Classifier
#clf = RandomForestClassifier()


#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


### Decision Tree, 

clf = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')

#param_grid = {}
#clf = GridSearchCV(dt, param_grid)
#clf = clf.fit(features, labels)
#print clf.best_estimator_

### AdaBoost Classifier
#clf = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
#          learning_rate=1.0, n_estimators=50, random_state=None)
#param_grid = {}
#clf = GridSearchCV(ada, param_grid)
#clf = clf.fit(features, labels)
#print clf.best_estimator_    
    
### K Nearest Neighbors Classifier
#clf = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
#           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
#           weights='uniform')
#param_grid = {}
#clf = GridSearchCV(knn, param_grid)
#clf = clf.fit(features, labels)
#print clf.best_estimator_

### Random Forest Classifier
#clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#            max_depth=None, max_features='auto', max_leaf_nodes=None,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=1, min_samples_split=2,
#            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
#            oob_score=False, random_state=None, verbose=0,
#            warm_start=False)
#param_grid = {}
#clf = GridSearchCV(rf, param_grid)
#clf = clf.fit(features, labels)
#print clf.best_estimator_


dump_classifier_and_data(clf, my_dataset, features_list)

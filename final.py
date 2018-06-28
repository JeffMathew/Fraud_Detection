import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score as score
from xgboost.sklearn import XGBClassifier
from xgboost import to_graphviz
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import pydot


#read database
df = pd.read_csv('PS_20174392719_1491204439457_log.csv')
#rename columns
df = df.rename(columns={'oldbalanceOrg':'oldBalanceOrig', 'newbalanceOrig':'newBalanceOrig','oldbalanceDest':'oldBalanceDest', 'newbalanceDest':'newBalanceDest'})


#only TRANSFER and CASH_OUT types show isFraud value = 1, so limiting data to that(refer Appendix A in report)
X = df.loc[(df.type == 'TRANSFER') | (df.type == 'CASH_OUT')]

targets = X['isFraud']
del X['isFraud']

#removing the features that don't affect the outcome as determined during data analysis (refer Appendix A thorugh F of report)
X = X.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis = 1)

X.loc[X.type == 'TRANSFER', 'type'] = 0 #assigning a numerical value in place of a string value so that algorithm can use it
X.loc[X.type == 'CASH_OUT', 'type'] = 1 #same as above
X.type = X.type.astype(int)

# when oldBalanceDest = newBalDest = 0, it is a strong indicator of fraud. Again, assigning a specific numeric value to those two columns in all database entries where the above condition is satisfied

X.loc[(X.oldBalanceDest == 0) & (X.newBalanceDest == 0) & (X.amount != 0),['oldBalanceDest', 'newBalanceDest']] = -1

# when oldBalOrig = newBalOrig = 0, it is a sign of non-fraud. Assigning a numeric value to those two columns.
X.loc[(X.oldBalanceOrig == 0) & (X.newBalanceOrig == 0) & (X.amount != 0),['oldBalanceOrig', 'newBalanceOrig']] = -2



feature_list = list(X.columns) 


#splitting the dataset into training and test sets
trainX, testX, trainTargets, testTargets = train_test_split(X, targets, test_size = 0.2,random_state = 42)






#RANDOM FOREST
rf = RandomForestClassifier(n_estimators = 10,max_depth = 3,random_state = 42) 
rf.fit(trainX, trainTargets) #training the model using the training set
predictions = rf.predict(testX) #making  predictions on the test data based on what it learned from training phase

print "Accuracy Score of Random Forest model is :-\n"
print score(testTargets,predictions) #comparing how many of the predicitons are same as actual true values. Best score is 1.00


tree = rf.estimators_[5] #arbitrarily taking the 5th of the 10 trees that were generated

# Export the image to a dot file
export_graphviz(tree, out_file = 'branch.dot', feature_names = feature_list)

# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('branch.dot')

# Write graph to a png file
graph.write_png('branch.png')#displaying the 5th tree as a png image







#XGBclassifier

clf = XGBClassifier(max_depth = 3,n_jobs = 4)
probabilities = clf.fit(trainX, trainTargets).predict_proba(testX)#training the model and making predictions on the test data

print "\nAccuracy Score of XGBoost model is:-\n"

print score(testTargets,probabilities[:,1].round())#comparing how many of the predictions are same as true values. Best score is 1.00

dot = to_graphviz(clf)
dot.format = 'png'
dot.render('xgbranch.gv',view = True)#visualise the strong learner tree as a png image.








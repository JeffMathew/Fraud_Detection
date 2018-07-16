from IPython import get_ipython
ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline')
import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix as confusion
from xgboost.sklearn import XGBClassifier
from xgboost import to_graphviz
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import pydot
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import average_precision_score

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import math
import scikitplot as skplt

from numpy import *
import matplotlib.pyplot as plt
from pylab import *
from string import ascii_uppercase






#IMPORT DATABASE

#read database
df = pd.read_csv('PS_20174392719_1491204439457_log.csv')
#rename columns
df = df.rename(columns={'oldbalanceOrg':'oldBalanceOrig', 'newbalanceOrig':'newBalanceOrig','oldbalanceDest':'oldBalanceDest', 'newbalanceDest':'newBalanceDest'})



#DATA ANALYSIS AND CLEANING

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




#Combining existing features to create a new one
X['errorBalanceOrig'] = X.newBalanceOrig + X.amount - X.oldBalanceOrig




feature_list = list(X.columns) 







#VISUALIZATION

limit = len(X)

def plotStrip(x, y, hue, figsize = (15, 20)): #Function to lay down the base detais of graph plot
    
    fig = plt.figure(figsize = figsize)
    colours = plt.cm.Set1(np.linspace(0, 1, 9))
    with sns.axes_style('ticks'):
        ax = sns.stripplot(x, y,hue = hue, jitter = 0.4, marker = '.',size = 4, palette = colours)
        ax.set_xlabel('')
        ax.set_xticklabels(['genuine', 'fraudulent'], size = 16)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)

        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles, ['Transfer', 'Cash out'], bbox_to_anchor=(1, 1),loc=3, borderaxespad=0, fontsize = 16);
    return ax



def cfmat(x,s): #Function to display Confusion Matrix
	columns = ['class %s' %(i) for i in list(ascii_uppercase)[0:len(np.unique(testTargets))]]
	
	confm = confusion(testTargets,x)
	df_cm = DataFrame(confm, index=columns, columns=columns)

	ax = sns.heatmap(df_cm, cmap='Oranges', annot=True)
	ax.set_title('Confusion Matrix of %s'%(s),size = 14)
	plt.show()








# Plot over ErrorBalOrig
limit = len(X)
ax = plotStrip(targets[:limit], -X.errorBalanceOrig[:limit], X.type[:limit],figsize = (15, 20))
ax.set_ylabel('- errorBalanceOrig', size = 16)
ax.set_title('Fraud and Non-Fraud Transactions Over ErrorBalOrig', size = 14);
plt.show()

#Plot over Amount

limit = len(X)
ax = plotStrip(targets[:limit], X.amount[:limit], X.type[:limit], figsize = (15, 20))
ax.set_ylabel('amount', size = 16)
ax.set_title('Fraud and Non-Fraud Transactions Over Amount', size = 14);
plt.show()





#MACHINE LEARNING ALGORITHM


#Splitting dataset into training and test sets
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2,random_state=42)
for train_index, test_index in sss.split(X, targets):
	#print("TRAIN:", train_index, "TEST:", test_index)
	trainX, testX = X.iloc[train_index], X.iloc[test_index]
	trainTargets, testTargets = targets.iloc[train_index], targets.iloc[test_index]


non_count = 0
fraud_count = 0
print "\nThe number of Frauds in test set are:-\n"
for i in testTargets:
	if i == 1:
		fraud_count = fraud_count + 1
	else:
		non_count = non_count + 1

print fraud_count

print "\n The number of non_frauds are:-\n"
print non_count




#RANDOM FOREST
rf = RandomForestClassifier(n_estimators = 10,max_depth = 7,random_state = 42) 
rf.fit(trainX, trainTargets) #training the model using the training set
predictions = rf.predict(testX) #making  predictions on the test data based on what it learned from training phase
probs = rf.predict_proba(testX)




print "Confusion Matrix of Random Forest \n"
print confusion(testTargets, predictions)
cfmat(predictions,'Random Forest')#Display Confusion Matrix

print "\nAUPRC score of Random Forest is:-\n"
print average_precision_score(testTargets,probs[:,1]) #Display the AUPRC score

skplt.metrics.plot_precision_recall(testTargets,probs)#Plot AUPRC Graph
plt.show()


tree = rf.estimators_[5] #arbitrarily taking the 5th of the 10 trees that were generated

# Export the image to a dot file
export_graphviz(tree, out_file = 'branch.dot', feature_names = feature_list)

# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('branch.dot')

# Write graph to a png file
graph.write_png('branch.png')#displaying the 5th tree as a png image





#XGBOOST

clf = XGBClassifier(max_depth = 3,n_jobs = 4)
probabilities = clf.fit(trainX, trainTargets).predict_proba(testX)#training the model and making predictions on the test data
preds = clf.predict(testX)

print "\nConfusion Matrix of XGBoost is:-\n"

print confusion(testTargets,preds)
cfmat(preds,'XGBoost')#Display Confusion Matrix

print "\nAUPRC score of XGBoost is:-\n"
print average_precision_score(testTargets,probabilities[:, 1])#Display the AUPRC score

skplt.metrics.plot_precision_recall(testTargets,probabilities)#Plot AUPRC graph
plt.show()

dot = to_graphviz(clf)
dot.format = 'png'
dot.render('xgbranch.gv',view = True)#visualise the strong learner tree as a png image.











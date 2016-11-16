import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import ml_metrics as metric
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

df= pd.read_csv('cs-training.csv')
def num_missing(x):
  return sum(x.isnull())

df.to_csv('mytraining.csv')
m=df.dropna()
sacuvaj=m.to_csv('deletedNans.csv')
m = m[m['age'] > 0] #Age=0 Deleted

m= m[m['NumberOfTime30-59DaysPastDueNotWorse'] < 96]
m= m[m['NumberOfTime60-89DaysPastDueNotWorse']<96]
m= m[m['NumberOfTimes90DaysLate']<96]

sample_size = len(m[m.SeriousDlqin2yrs == 1]) # get the total count of low-frequency group

small_indices = m[m.SeriousDlqin2yrs == 1].index
small_sample = m.loc[small_indices]

big_indices = m[m.SeriousDlqin2yrs == 0].index
random_indices = np.random.choice(big_indices, sample_size, replace=False) # use the low-frequency group count to randomly sample from high-frequency group
big_sample = m.loc[random_indices]

# Merging all the low-frequency group sample and the new (randomly selected) high-frequency sample together
merged_sample = pd.concat([small_sample, big_sample], ignore_index=True)
#print merged_sample

def classification_model(model, m, predictors, outcome):
  #Fit the model:
  model.fit(m[predictors],m[outcome])

  #Make predictions on training set:
  predictions = model.predict(m[predictors])

  #Print accuracy
  accuracy = metrics.accuracy_score(predictions,m[outcome])
  print "Accuracy : %s" % "{0:.3%}".format(accuracy)

  auc=metric.auc(predictions, m[outcome])
  print "Auc : %s" % "{0:.3%}".format(auc)

  recall = metrics.recall_score(predictions,m[outcome])
  print "Recall : %s" % "{0:.3%}".format(recall)


  #Fit the model again so that it can be refered outside the function:
  model.fit(m[predictors],m[outcome])

#outcome_var = 'SeriousDlqin2yrs'
#model = LogisticRegression()#
#predictor_var = ['RevolvingUtilizationOfUnsecuredLines','age','NumberOfTime30-59DaysPastDueNotWorse','DebtRatio','MonthlyIncome','NumberOfOpenCreditLinesAndLoans','NumberOfTimes90DaysLate','NumberRealEstateLoansOrLines','NumberOfTime60-89DaysPastDueNotWorse','NumberOfDependents']
#classification_model(model, merged_sample,predictor_var,outcome_var)



#outcome_var = 'SeriousDlqin2yrs'
#model = DecisionTreeClassifier(criterion='gini') #Accuracy : 76.991%
#Auc : 77.035%
##Recall : 75.953%#
#Cross-Validation Score : 66.065%
#predictor_var = ['RevolvingUtilizationOfUnsecuredLines','age','NumberOfTime30-59DaysPastDueNotWorse','DebtRatio','MonthlyIncome','NumberOfOpenCreditLinesAndLoans','NumberOfTimes90DaysLate','NumberRealEstateLoansOrLines','NumberOfTime60-89DaysPastDueNotWorse','NumberOfDependents']
#classification_model(model, merged_sample,predictor_var,outcome_var)


#outcome_var = 'SeriousDlqin2yrs'
##model = RandomForestClassifier(max_depth=5)
#predictor_var = ['RevolvingUtilizationOfUnsecuredLines','age','NumberOfTime30-59DaysPastDueNotWorse','DebtRatio','MonthlyIncome','NumberOfOpenCreditLinesAndLoans','NumberOfTimes90DaysLate','NumberRealEstateLoansOrLines','NumberOfTime60-89DaysPastDueNotWorse','NumberOfDependents']
#classification_model(model, merged_sample,predictor_var,outcome_var) #Accuracy : 98.253% #Cross-Validation Score : 70.484%


testfile = pd.read_csv('cs-test.csv')
testfile.info()
deleteMisValues= testfile.fillna(method='ffill')
deleteMisValues.info()

outcome_var = 'SeriousDlqin2yrs'
model = RandomForestClassifier(max_depth=8, min_samples_leaf=9)
predictor_var = ['RevolvingUtilizationOfUnsecuredLines','age','NumberOfTime30-59DaysPastDueNotWorse','DebtRatio','MonthlyIncome','NumberOfOpenCreditLinesAndLoans','NumberOfTimes90DaysLate','NumberRealEstateLoansOrLines','NumberOfTime60-89DaysPastDueNotWorse','NumberOfDependents']
classification_model(model, merged_sample,predictor_var,outcome_var)

predictions = model.predict(deleteMisValues[predictor_var])
print(predictions)

np.savetxt('CRtest4.csv', predictions, delimiter=',')
np.genfromtxt('CRtest4.csv', delimiter=',')

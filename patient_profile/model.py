#-*- coding: utf-8 -*-

import pandas as pd
from sklearn import svm
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tabulate import tabulate



# read and convert
#data_x = pd.read_csv("/mnt/4TB1/onj/dataset/v0/X.csv")
#data_y = pd.read_csv("/mnt/4TB1/onj/dataset/v0/Y.csv")

data_x = pd.read_csv('D:\Work\보건복지부과제\ONJ\onj\X.csv') 
data_y = pd.read_csv('D:\Work\보건복지부과제\ONJ\onj\Y.csv')



# additionally dropping columns
except_list_2 = ['Pain', 'Swelling', 'Exposure', 'Abscess', 'Paresthesia', 'Mobility', 'Oroantral Fistula', 'Orocutaneous fistula']
except_list_3 = ['Low dosing', 'High dosing', 'Exposure']
#data_x = data_x.drop(columns=except_list_2)
#data_x = data_x.drop(columns=except_list_3)



# split and squeeze
len_data_x = len(data_x)
len_data_y = len(data_y)


train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, shuffle=False, random_state=0)

train_x = train_x.drop(train_x.columns[0],axis=1)
test_x = test_x.drop(test_x.columns[0],axis=1)
train_y = train_y.drop(train_y.columns[0],axis=1)
test_y = test_y.drop(test_y.columns[0],axis=1)

train_y = train_y.squeeze()
test_y = test_y.squeeze()



# train
classifier = svm.SVC(kernel='linear', C=2.2)
classifier.fit(train_x, train_y)


supportVectors = classifier.support_vectors_
supportVectorIndices = classifier.support_
supportVectorCount = classifier.n_support_




# test
testPrediction = classifier.predict(test_x)


confusionMatrix = confusion_matrix(test_y, testPrediction)
print("Confusion Matrix: \n", confusionMatrix)
print("Accuracy of the Classifier: %.2f" %(100*accuracy_score(test_y, testPrediction)),"%")
print("Precision value of the Positive Class ('Normal' Class): %.2f" %(100*precision_score(test_y, testPrediction,  average="binary")), "%")
print("Recall value of the Positive Class ('Normal' Class): %.2f" %(100*recall_score(test_y, testPrediction,  average="binary")), "%")

#Classification Report for each class:
target_names = ['Normal', 'Abnormal']
print("Precision and Recall for each class: \n",classification_report(test_y, testPrediction, target_names=target_names))



# visualizing confusion matrix
import seaborn as sns
fig, ax= plt.subplots(figsize=(8,6))
sns.heatmap(confusionMatrix, annot=True, linewidths=.5, xticklabels=True, yticklabels=True)
ax.xaxis.set_ticklabels(['Normal', 'Abnormal'], fontsize=12); ax.yaxis.set_ticklabels(['Normal', 'Abnormal'], fontsize=12)
ax.set_title('Confusion Matrix', fontsize=20)
ax.set_xlabel('Predicted Labels', fontsize=14)
ax.set_ylabel('True Labels', fontsize=14)

#plt.show()


# print auc & coef
print("area under curve (auc): ", roc_auc_score(test_y, testPrediction))

coef = classifier.coef_
feature_coef = pd.DataFrame()
feature_coef['feature'] = train_x.columns
coef = coef.reshape(-1,)
feature_coef['coef'] = coef
print(tabulate(feature_coef, headers='keys', tablefmt='fancy_outline', showindex=True))

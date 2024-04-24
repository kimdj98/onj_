import pandas as pd
from sklearn.utils import shuffle
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer



data = pd.read_excel('D:\Work\보건복지부과제\ONJ\onj\ONJ_AI임상정보.xlsx')
data.to_csv('D:\Work\보건복지부과제\ONJ\onj\ONJ_patient_profile.csv')
data = pd.read_csv('D:\Work\보건복지부과제\ONJ\onj\ONJ_patient_profile.csv')


###### change data to numbers ######
data['true'] = 0
data['true'] = np.where(data.iloc[:,[15]]!='O', 0, 1)
data.iloc[:,[15]] = data['true'] 

data = data.to_numpy()


###### data ######
data = np.delete(data, (1, 2, 16, 17, 18, 19, 20, 21, 22, 52, 60, 61, 62, 63, 64,  65,  74,  75,  76, 77, 78,  79, 80,  81, 82, 83,  84, 85, 86, 87), axis = 1)
data = np.delete(data, 0, axis=1)
data = np.delete(data, (0,1), axis=0)
data = data.astype(np.float64)

data = shuffle(data)




###### data_x ######
data_x = np.delete(data, 12, axis = 1)

for i in range(len(data_x)):
    if data_x[i][27] == 9999 and data_x[i][28] == 9999:
        data_x[i][27] = 0
        data_x[i][28] = 0
        data_x[i][29] = 1



###### data_y ######
data_y = data[:,12]
data_y = data_y.astype(np.float64)



###### change 9999 to NaN ######
data_x = np.where(data_x==9999, np.nan, data_x)


###### delete column with too many nan ###### 
ratio = 0.5
for i in reversed(range(data_x.shape[1])):
    count = np.count_nonzero(np.isnan(data_x[:,i]))
    if count > ratio*len(data_x):
        data_x = np.delete(data_x, i, axis=1)


####### replace 8888 to 0 #######
data_x = np.where(data_x==8888, 0, data_x)


###### MICE ######
## documentation : https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn.impute.IterativeImputer ##
imputer = IterativeImputer(max_iter=20)
data_x = imputer.fit_transform(data_x)

data_x = np.where(data_x < 0, abs(data_x), data_x)



####### save data #######
data_x = pd.DataFrame(data_x)
data_y = pd.DataFrame(data_y)


#data_x.to_csv("/mnt/4TB1/onj/dataset/v0/X.csv")     
#data_y.to_csv("/mnt/4TB1/onj/dataset/v0/Y.csv")     # 나중에 상대경로로 수정
data_x.to_csv('D:\Work\보건복지부과제\ONJ\onj\X.csv') 
data_y.to_csv('D:\Work\보건복지부과제\ONJ\onj\Y.csv')



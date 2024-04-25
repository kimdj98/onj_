import pandas as pd
from sklearn.utils import shuffle
import numpy as np
from sklearn.impute import IterativeImputer


# read data 
data = pd.read_excel('D:\Work\보건복지부과제\ONJ\onj\ONJ_AI임상정보.xlsx')
data.to_csv('D:\Work\보건복지부과제\ONJ\onj\ONJ_patient_profile.csv')
data = pd.read_csv('D:\Work\보건복지부과제\ONJ\onj\ONJ_patient_profile.csv')



# change symbol to number
data['true'] = 0
data['true'] = np.where(data['ONJ 진단여부']!='O', 0, 1)
data['ONJ 진단여부'] = data['true'] 
data = data.drop(columns='true')



# shuffle and reset row index
data = shuffle(data)
data.reset_index(drop=True, inplace=True)



# data_x
column_list = ['등록번호', '코드번호', 'AGE', 'SEX', 'HIG', 'WEI', 'BMI',
       'BMI_R', 'SBP', 'DBP', 'LQ', 'SM', 'DR', 'PM', 'ONJ 진단여부', 'stage',
       'diagnosis date', 'ONJ 완치일자', 'completion date', '수술날짜 1', '수술날짜2',
       '수술날짜3', 'Osteoporosis', 'Cancer', 'Hypertension', 'Diabetes Mellitus',
       'CVA', 'Osteoarthritis', 'Rheumatic Arthritis', 'renal failure',
       'hyperlipidemia', 'hypothyroidism', 'Hyperthyroidism', 'Low dosing',
       'High dosing', 'Combination (BP-Deno)', 'Combination (Deno-BP)', 'PO',
       'IV', '모름', '복용개월', 'Prolia', 'Xgeva', 'Alendronate', 'Ibandronate',
       'Zoledronate', 'Risedronate', 'steroid', '류마티스약물', 'Romosozumab', '기타',
       'Unknown Antiresorptives', 'Implant Associated Case 여부', 'extraction',
       'denture', 'torus', 'periapical lesion', 'Periodontitis', 'Spontaneous',
       'UL', 'UC', 'UR', 'LL', 'LC', 'LR', 'Pain', 'Swelling', 'Exposure',
       'Abscess', 'Paresthesia', 'Mobility', 'Oroantral Fistula',   
       'Orocutaneous fistula', ' Drug cessation 개월수', 'Drug holiday 개월 수',
       'Anesthesia', 'BMP', 'PRF', 'PTH', 'Early Wide resection',   
       'PTH_Wide resection', 'PTH_Sequestration', 'Sequestration_Only',
       'Conservative', '재발환자 유무']


except_list = ['등록번호', '코드번호', 'ONJ 진단여부', 'stage', 'diagnosis date', 'ONJ 완치일자', 'completion date', '수술날짜 1', '수술날짜2',
       '수술날짜3', 'Unknown Antiresorptives', 'UL', 'UC', 'UR', 'LL', 'LC', 'LR', ' Drug cessation 개월수', 'Drug holiday 개월 수',
       'Anesthesia', 'BMP', 'PRF', 'PTH', 'Early Wide resection', 'PTH_Wide resection', 'PTH_Sequestration', 'Sequestration_Only',
       'Conservative', '재발환자 유무']



data_x = data[column_list]
data_x = data_x.drop(columns=except_list)
data_x = data_x.astype(dtype=np.float64)


index = data_x[(data_x['PO']==9999)&(data_x['IV']==9999)].index
for i in index :
    data_x.at[i,'PO'] = 0
    data_x.at[i,'IV'] = 0
    data_x.at[i,'모름'] = 1



# data_y 
data_y = data['ONJ 진단여부']
data_y = data_y.astype(dtype=np.float64)



# change 9999 to NaN
data_x = data_x.replace(to_replace=9999, value=np.NaN)



# replace 8888 to 0
data_x = data_x.replace(to_replace=8888, value=0)



# delete column with too many nan (over 50%)
ratio = 0.5
data_x = data_x.dropna(thresh=ratio*len(data_x), axis=1)    # thresh = num of existing data



x_column_name = data_x.columns

# MICE 
imputer = IterativeImputer(max_iter=20)
data_x = imputer.fit_transform(data_x)

data_x[data_x < 0] = 0





####### save data #######
data_x = pd.DataFrame(data_x)
data_y = pd.DataFrame(data_y)

data_x.columns = x_column_name


data_x.to_csv('D:\Work\보건복지부과제\ONJ\onj\X.csv') 
data_y.to_csv('D:\Work\보건복지부과제\ONJ\onj\Y.csv')

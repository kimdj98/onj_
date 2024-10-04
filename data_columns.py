import pandas as pd
from sklearn.utils import shuffle
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from fancyimpute import IterativeImputer as MICE
import os 


column_list = ['pt_H',  'ONJ_DIA_AGE', 'SEX', 'HIG', 'WEI', 'BMI', 'BMI_R', 'SBP', 'DBP', 'LQ', 'SM', 'DR', 'PM', 'ONJ_DIA_CK',
              'ONJ_DIA_ST', 'ONJ_DIA_DD', 'ONJ_CP_DD', 'CON_CP_DD', 'SUR_DD_1', 'SUR_DD_2', 'SUR_DD_3', 'CP_DD_un', 'CP', 'ONJ_OST', 'ONJ_CC', 'MH_HTN', 
              'MH_DM', 'MH_CVA', 'MH_OA', 'MH_RART', 'MH_RF', 'MH_HYP', 'MH_HYPO', 'MH_HYPE', 'MH_CC', 'PMH_L', 'PMH_H', 'PMH_BD', 'PMH_DB', 'PMH_PO',      
              'PMH_IVSC', 'PMH_CK', 'PMH_MM', 'PMH_MM_un', 'PMH_t_PRO', 'PMH_t_XG', 'PMH_t_ALEN', 'PMH_t_BAN', 'PMH_t_ZOLE', 'PMH_t_RISED', 'PMH_t_STE',     
              'PMH_t_RHEUM', 'PMH_t_ROMO', 'PMH_t_SERM',  'PMH_C_ANTI', 'PMH__ANT', 'PMH_un', 'OR_IMP', 'OR_EXT', 'OR_DEN', 'OR_TOR', 'OR_RCT',  
              'OR_PE', 'OR_SPO', 'ONJ_SI_UL', 'ONJ_SI_UC', 'ONJ_SI_UR', 'ONJ_SI_LL', 'ONJ_SI_LC', 'ONJ_SI_LR', 'OE_PA', 'OE_SW', 'OE_BEX', 'OE_PUS',
              'OE_PAR', 'OE_MOB', 'OE_IN_FIS', 'OE_EX_FIS', 'SUR_DC_MM', 'SUR_RE_DC_MM', 'SUR_DC_CON', 'SUR_AN', 'SUR_BMP', 'SUR_PRF', 'SU_PTH',  
              'SUR',  'SUR_RE']
              ## 업데이트본에서 빠진 변수들
              # 'pt_H_ID', 'pt_CODE', 'PMH_OTHERS', 'Early Wide resection', 'PTH_Wide resection', 'PTH_Sequestration', 'Sequestration_Only', 'Conservative',

except_list = ['pt_H',  'ONJ_DIA_CK' , 'ONJ_DIA_ST', 'ONJ_DIA_DD', 'ONJ_CP_DD', 'CON_CP_DD', 'SUR_DD_1', 'SUR_DD_2', 'SUR_DD_3', 
              'CP_DD_un' , 'CP' , 'ONJ_SI_UL', 'ONJ_SI_UC', 'ONJ_SI_UR', 'ONJ_SI_LL', 'ONJ_SI_LC', 'ONJ_SI_LR' , 'SUR_DC_MM', 'SUR_RE_DC_MM', 
              'SUR_DC_CON', 'SUR_AN', 'SUR_BMP', 'SUR_PRF', 'SU_PTH', 'SUR',  'SUR_RE'] # 'PMH_un'는 이제 뺄 이유가 없음
              # 'pt_H_ID', 'pt_CODE' , 'Early Wide resection', 'PTH_Wide resection', 'PTH_Sequestration', 'Sequestration_Only', 'Conservative',



def onj_convert(data):
       data['true'] = 0
       data['true'] = np.where(data['ONJ_DIA_CK']!='O', 0, 1)
       data['ONJ_DIA_CK'] = data['true'] 
       data = data.drop(columns='true')

       return data



def data_make(data):
       

       #data = shuffle(data)
       data.reset_index(drop=True, inplace=True)





       '''
       ###### data ######
       index = data[data['Month of Taking'].str.contains(r"\+")].index
       for i in index :
              data.at[i,'Month of Taking'] = data.at[i,'Month of Taking'].split('+')[0]
       '''
       # 이 부분 일단 프로세싱 보류 ->> 두 열을 그냥 제외해뒀는데 나중에 저 아래 drop 지우기  >> 이제 이거 필요없어짐
       
       

       



           
       ###### data_x ######
       data_x = data[column_list]
       data_x = data_x.drop(columns=except_list)
       data_x = data_x.astype(dtype=np.float64)
       
       
       ###### medicine name to 'yes==1' ######
       # def 밖에 해둠 이거 필요없음
       '''
       data[data['PMH_OTHERS'].dtype()==str] = 1
       data['PMH_OTHERS'] = data['PMH_OTHERS'].apply(lambda x: 1 if x != 0 else x)
       index = data[data['PMH_OTHERS']!=0].index
       for i in index :
              data.at[i,'PMH_OTHERS'] = 1
       data['PMH_OTHERS'] = np.where(data['PMH_OTHERS']!=0, 0, 1)
       # 코딩으로 해보려고 했는데 일단은 아직 안돼서 직접 데이터 고쳐놓음. 나중에 코드 완성해두기
       '''
       '''
       data_x['others'] = 1

       data_x.loc[(data_x['PMH_OTHERS'] == 0), 'others'] = 0
       data_x['PMH_OTHERS'] = data_x['others']
       data_x = data_x.drop(columns='others')
       '''





       ###### PO, IV, Dont know ######
       index = data_x[(data_x['PMH_PO']==9999)&(data_x['PMH_IVSC']==9999)].index
       for i in index :
              data_x.at[i,'PMH_PO'] = 0
              data_x.at[i,'PMH_IVSC'] = 0
              data_x.at[i,'PMH_CK'] = 1



       ''' 이거는  mice 하고 나서 하는 걸로 순서바꾸기
       ###### 발병요인 합치기 ######
       # 없음==0 , IMP==1 , EXT==2 , DEN==3 , TOR==4 , RCT==5 , PE==6 , SPO==7
       # 일단 변수 정리부터
       or_list = ['OR_IMP', 'OR_EXT', 'OR_DEN', 'OR_TOR', 'OR_RCT', 'OR_PE']
       data_x['count'] = data_x['OR_IMP'] + data_x['OR_EXT'] + data_x['OR_DEN'] + data_x['OR_TOR'] + data_x['OR_RCT'] + data_x['OR_PE']
       data_x.loc[(data_x['count'] >  1) , 'OR_SPO'] = 1
       data_x.loc[(data_x['count'] >  1) , or_list] = 0
       data_x = data_x.drop(columns='count')

       data_x['OR'] = 0

       data_x.loc[(data_x['OR_IMP'] ==  1) , 'OR'] = 1
       data_x.loc[(data_x['OR_EXT'] ==  1) , 'OR'] = 2
       data_x.loc[(data_x['OR_DEN'] ==  1) , 'OR'] = 3
       data_x.loc[(data_x['OR_TOR'] ==  1) , 'OR'] = 4
       data_x.loc[(data_x['OR_RCT'] ==  1) , 'OR'] = 5
       data_x.loc[(data_x['OR_PE'] ==  1) , 'OR'] = 6
       data_x.loc[(data_x['OR_SPO'] ==  1) , 'OR'] = 7

       data_x = data_x.drop(columns=or_list)
       data_x = data_x.drop(columns='OR_SPO')
       '''



       ###### data_y ######
       data_y = data['ONJ_DIA_CK']
       data_y = data_y.astype(dtype=np.float64)




       ###### change 9999 to NaN ######
       data_x = data_x.replace(to_replace=9999, value=np.NaN)



       ####### replace 8888 to 0 #######
       data_x = data_x.replace(to_replace=8888, value=0)



       ##### delete column with too many nan ###### 
       #ratio = 0.5
       #data_x = data_x.dropna(thresh=ratio*len(data_x), axis=1)    # thresh = num of existing data



       ##### delete data too close to label #####
       except_list_2 = ['OE_PA', 'OE_SW', 'OE_BEX', 'OE_PUS', 'OE_PAR', 'OE_MOB', 'OE_IN_FIS', 'OE_EX_FIS']
       data_x = data_x.drop(columns=except_list_2)


       return data_x , data_y



def MICE(df):      ### 인코딩은 범주형 변수 몇개만 하고 바로 mice 돌려버리기
       ### 지금 연속형이랑 범주형이 섞여있어서 잘 안되는 걸 수도 있음. BMI만 떼다가 돌려보고 다시 끼워서 한번 더 돌려봅시다
       #conti_list = ['SEX' , 'ONJ_DIA_AGE' , 'HIG' , 'WEI' , 'BMI']   ### 결측치가 없어서 sex도 뺌. 나중에 디코딩하기 편하게
       #data = df.drop(columns=conti_list)


       ## 차라리 범주형 변수인 것들만 골라내기
       cati_list = ['SM' , 'DR']       # 뭐야 겨우 이 세놈 때문에 그 고생을 한거?

       ## onehot encoding
       data_encoded = pd.get_dummies(df , columns=cati_list , dtype=int)
       

       ## 인코딩된 값들 중에서 nan 살리기
       # 일단 원래 columns들 소환 #
       data_encoded['SM'] = data_x['SM']
       data_encoded['DR'] = data_x['DR']
       column_encoded = data_encoded.columns

       data_encoded.loc[data_encoded['SM'].isna() , 'SM_1.0'] = np.nan
       data_encoded.loc[data_encoded['SM'].isna() , 'SM_2.0'] = np.nan
       data_encoded.loc[data_encoded['SM'].isna() , 'SM_3.0'] = np.nan
       data_encoded.loc[data_encoded['SM'].isna() , 'SM_4.0'] = np.nan

       data_encoded.loc[data_encoded['DR'].isna() , 'DR_0.0'] = np.nan
       data_encoded.loc[data_encoded['DR'].isna() , 'DR_1.0'] = np.nan
       data_encoded.loc[data_encoded['DR'].isna() , 'DR_2.0'] = np.nan


       ## 결측치 대체
       imputer = IterativeImputer(max_iter=50, random_state=0)               ## early stop이 발생하지 않도록 max_iter를 충분히
       imputer.fit(data_encoded)
       im_test = imputer.transform(data_encoded)
       

       

       # 데이터프레임으로 만들기, 정수화 #
       test_last = pd.DataFrame(im_test, columns=column_encoded)
       yeah = ['PMH_MM', 'PMH_MM_un', 'SM_1.0', 'SM_2.0', 'SM_3.0', 'SM_4.0', 'DR_0.0', 'DR_1.0', 'DR_2.0']
       test_last[yeah] = test_last[yeah].round(0)

       ## BMI_R은 인코딩해서 mice 하지 말고 그냥 BMI로 계산해서 넣기 ##
       test_last.loc[(test_last['BMI'] < 18.5) , 'BMI_R'] = 0
       test_last.loc[(test_last['BMI'] >= 18.5) &  (test_last['BMI'] < 23), 'BMI_R'] = 1
       test_last.loc[(test_last['BMI'] >= 23) &  (test_last['BMI'] < 25) , 'BMI_R'] = 2
       test_last.loc[(test_last['BMI'] >= 25) &  (test_last['BMI'] < 30) , 'BMI_R'] = 3
       test_last.loc[(test_last['BMI'] >= 30) , 'BMI_R'] = 4



       ## 디코딩
       test_last.loc[(test_last['SM_1.0'] ==  1) , 'SM'] = 1
       test_last.loc[(test_last['SM_2.0'] ==  1) , 'SM'] = 2
       test_last.loc[(test_last['SM_3.0'] ==  1) , 'SM'] = 3
       test_last.loc[(test_last['SM_4.0'] ==  1) , 'SM'] = 4

       test_last.loc[(test_last['DR_0.0'] ==  1) , 'DR'] = 0
       test_last.loc[(test_last['DR_1.0'] ==  1) , 'DR'] = 1
       test_last.loc[(test_last['DR_2.0'] ==  1) , 'DR'] = 2

       # drop
       yeah = ['SM_1.0', 'SM_2.0', 'SM_3.0', 'SM_4.0', 'DR_0.0', 'DR_1.0', 'DR_2.0']
       test_last = test_last.drop(columns=yeah)

       return test_last







#path = 'D:/Work/보건복지부과제/ONJ/onj/inAndOut_onj'
path = 'F:/노트북/Work/보건복지부과제/ONJ/onj/inAndOut_onj'
os.chdir(path)

df = pd.read_excel('이대목동+이대서울_241002.xlsx', names= column_list)
df.to_csv('ONJ_patient_clinical.csv')
df = pd.read_csv('ONJ_patient_clinical.csv', index_col=0)

#### 1차 전처리 ####
[data_x , data_y] = data_make(df)
#data_x.to_csv(path + '/X_EW.csv') 
#data_y.to_csv(path + '/Y_EW.csv')


#### MICE ####
data_x = MICE(data_x)
#data_x.to_csv(path + '/X_EW_mice_2.csv')




###### 발병요인 합치기 ######
# 없음==0 , IMP==1 , EXT==2 , DEN==3 , TOR==4 , RCT==5 , PE==6 , SPO==7
# 일단 변수 정리부터
or_list = ['OR_IMP', 'OR_EXT', 'OR_DEN', 'OR_TOR', 'OR_RCT', 'OR_PE']
data_x['count'] = data_x['OR_IMP'] + data_x['OR_EXT'] + data_x['OR_DEN'] + data_x['OR_TOR'] + data_x['OR_RCT'] + data_x['OR_PE']
data_x.loc[(data_x['count'] >  1) , 'OR_SPO'] = 1
data_x.loc[(data_x['count'] >  1) , or_list] = 0
data_x = data_x.drop(columns='count')

data_x['OR'] = 0

data_x.loc[(data_x['OR_IMP'] ==  1) , 'OR'] = 1
data_x.loc[(data_x['OR_EXT'] ==  1) , 'OR'] = 2
data_x.loc[(data_x['OR_DEN'] ==  1) , 'OR'] = 3
data_x.loc[(data_x['OR_TOR'] ==  1) , 'OR'] = 4
data_x.loc[(data_x['OR_RCT'] ==  1) , 'OR'] = 5
data_x.loc[(data_x['OR_PE'] ==  1) , 'OR'] = 6
data_x.loc[(data_x['OR_SPO'] ==  1) , 'OR'] = 7

data_x = data_x.drop(columns=or_list)
data_x = data_x.drop(columns='OR_SPO')



### 최종 데이터 저장 ###
## final in : '이대목동+이대서울_241002.xlsx' >> out : X_EW.csv, Y_EW.csv (이제 나머지 엑셀은 다 지워도 됨)
data_x.to_csv(path + '/X_EW.csv') 
data_y.to_csv(path + '/Y_EW.csv')





''' 이거 이제 필요없음
#data = pd.read_excel('ONJ_AI임상정보.xlsx')
df_1 = pd.read_excel('ONJ_AI임상정보.xlsx', sheet_name = '이대목동병원')            # 이대목동병원 == 1
df_2 = pd.read_excel('ONJ_AI임상정보.xlsx', sheet_name = '이대서울병원')            # 이대서울병원 == 2
#df_2 = onj_convert(df_2)                                                           # 이미 convert 되어서 업데이트

#OTHERS 아예 제외돼서 왔음
#df_1['others'] = 1
#df_1.loc[(df_1['PMH_OTHERS'] == 0), 'others'] = 0
#df_1['PMH_OTHERS'] = df_1['others']
#df_1 = df_1.drop(columns='others')


df_2['others'] = 1
df_2.loc[(df_2['PMH_OTHERS'] == 0), 'others'] = 0
df_2['PMH_OTHERS'] = df_2['others']
df_2 = df_2.drop(columns='others')



df_3 = pd.concat([df_1, df_2]).reset_index(drop=True)                             # 목동 & 서울 == 3

df_1.to_csv('ONJ_patient_clinical_EW.csv')
df_2.to_csv('ONJ_patient_clinical_EWBS.csv')
df_3.to_csv('ONJ_patient_clinical_EW+EWBS.csv')

#print(df_1.iloc[0,52])






df_1 = pd.read_csv('ONJ_patient_clinical_EW.csv', index_col=0)
df_2 = pd.read_csv('ONJ_patient_clinical_EWBS.csv', index_col=0)
df_3 = pd.read_csv('ONJ_patient_clinical_EW+EWBS.csv', index_col=0)




[data_x_1 , data_y_1] = data_make(df_1)
[data_x_2 , data_y_2] = data_make(df_2)
[data_x_3 , data_y_3] = data_make(df_3)


data_x_1.to_csv(path + '/X_EW.csv') 
data_y_1.to_csv(path + '/Y_EW.csv')
data_x_2.to_csv(path + '/X_EWBS.csv') 
data_y_2.to_csv(path + '/Y_EWBS.csv')
data_x_3.to_csv(path + '/X_EW+EWBS.csv') 
data_y_3.to_csv(path + '/Y_EW+EWBS.csv')

'''


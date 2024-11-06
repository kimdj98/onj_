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


column_list = ['pt_H', 'pt_CODE', 'ONJ_DIA_AGE', 'SEX', 'HIG', 'WEI', 'BMI', 'BMI_R', 'SBP', 'DBP', 'LQ', 'SM', 'DR', 'PM', 'ONJ_DIA_CK',
              'ONJ_DIA_ST', 'ONJ_DIA_DD', 'ONJ_CP_DD', 'CON_CP_DD', 'SUR_DD_1', 'SUR_DD_2', 'SUR_DD_3', 'CP_DD_un', 'CP', 'ONJ_OST', 'ONJ_CC', 'MH_HTN', 
              'MH_DM', 'MH_CVA', 'MH_OA', 'MH_RART', 'MH_RF', 'MH_HYP', 'MH_HYPO', 'MH_HYPE', 'MH_CC', 'PMH_L', 'PMH_H', 'PMH_BD', 'PMH_DB', 'PMH_PO',      
              'PMH_IVSC', 'PMH_CK', 'PMH_MM', 'PMH_MM_un', 'PMH_t_PRO', 'PMH_t_XG', 'PMH_t_ALEN', 'PMH_t_BAN', 'PMH_t_ZOLE', 'PMH_t_RISED', 'PMH_t_STE',     
              'PMH_t_RHEUM', 'PMH_t_ROMO', 'PMH_t_SERM',  'PMH_C_ANTI', 'PMH__ANT', 'PMH_un', 'OR_IMP', 'OR_EXT', 'OR_DEN', 'OR_TOR', 'OR_RCT',  
              'OR_PE', 'OR_SPO', 'ONJ_SI_UL', 'ONJ_SI_UC', 'ONJ_SI_UR', 'ONJ_SI_LL', 'ONJ_SI_LC', 'ONJ_SI_LR', 'OE_PA', 'OE_SW', 'OE_BEX', 'OE_PUS',
              'OE_PAR', 'OE_MOB', 'OE_IN_FIS', 'OE_EX_FIS', 'SUR_DC_MM', 'SUR_RE_DC_MM', 'SUR_DC_CON', 'SUR_AN', 'SUR_BMP', 'SUR_PRF', 'SU_PTH',  
              'SUR',  'SUR_RE']
              # 'pt_H_ID', 'PMH_OTHERS', 'Early Wide resection', 'PTH_Wide resection', 'PTH_Sequestration', 'Sequestration_Only', 'Conservative',

except_list = ['pt_H',  'pt_CODE', 'ONJ_DIA_CK' , 'ONJ_DIA_ST', 'ONJ_DIA_DD', 'ONJ_CP_DD', 'CON_CP_DD', 'SUR_DD_1', 'SUR_DD_2', 'SUR_DD_3', 
              'CP_DD_un' , 'CP' , 'ONJ_SI_UL', 'ONJ_SI_UC', 'ONJ_SI_UR', 'ONJ_SI_LL', 'ONJ_SI_LC', 'ONJ_SI_LR' , 'SUR_DC_MM', 'SUR_RE_DC_MM', 
              'SUR_DC_CON', 'SUR_AN', 'SUR_BMP', 'SUR_PRF', 'SU_PTH', 'SUR',  'SUR_RE' , 'PMH_MM_un']
              # 'pt_H_ID' , 'Early Wide resection', 'PTH_Wide resection', 'PTH_Sequestration', 'Sequestration_Only', 'Conservative', 'PMH_un'



def onj_convert(df):                                                    ## onj 여부를 기호로 표시한 걸 숫자로 바꾸는 함수.                                                  
       df['true'] = 0                                                   ## 최근 데이터에는 애초에 숫자로 되어있기 때문에 필요없음
       df['true'] = np.where(df['ONJ_DIA_CK']!='O', 0, 1)
       df['ONJ_DIA_CK'] = df['true'] 
       df = df.drop(columns='true')

       return df


def start(df):
       df = df.drop([0, 1]).reset_index(drop=True)                       ## 필요한 행만 남김
       df.columns = column_list                                          ## column name 정리
       df.to_excel('trash.xlsx')                                         ## 일단 엑셀로 저장
       df = pd.read_excel('trash.xlsx', names= column_list)              ## 다시 부름
       df.to_csv('ONJ_patient_clinical.csv')                             ## csv로 저장
       df = pd.read_csv('ONJ_patient_clinical.csv', index_col=0)         ## 다시 부름
                                                                         ## 복잡한 거 아는데 되는 코드니까 그냥 진행..
       return df




def data_make(data):
       
       data = shuffle(data)
       data.reset_index(drop=True, inplace=True)
       
           
       ###### data_x ######
       data_x = data[column_list]
       data_x = data_x.drop(columns=except_list)
       data_x = data_x.astype(dtype=np.float64)
       

    ###### PO, IV, Dont know ######
    index = data_x[(data_x["PMH_PO"] == 9999) & (data_x["PMH_IVSC"] == 9999)].index
    for i in index:
        data_x.at[i, "PMH_PO"] = 0
        data_x.at[i, "PMH_IVSC"] = 0
        data_x.at[i, "PMH_CK"] = 1
       ###### PO, IV, Dont know ######
       index = data_x[(data_x['PMH_PO']==9999)&(data_x['PMH_IVSC']==9999)].index
       for i in index :
              data_x.at[i,'PMH_PO'] = 0
              data_x.at[i,'PMH_IVSC'] = 0
              data_x.at[i,'PMH_CK'] = 1


    ###### data_y ######
    data_y = data["ONJ_DIA_CK"]
    data_y = data_y.astype(dtype=np.float64)
       ###### data_y ######
       data_y = data['ONJ_DIA_CK']
       data_y = data_y.astype(dtype=np.float64)


    ###### change 9999 to NaN ######
    data_x = data_x.replace(to_replace=9999, value=np.NaN)
       ###### change 9999 to NaN ######
       data_x = data_x.replace(to_replace=9999, value=np.NaN)


       ####### replace 8888 to 0 #######
       data_x = data_x.replace(to_replace=8888, value=0)




    ##### delete data too close to label #####
    except_list_2 = ["OE_PA", "OE_SW", "OE_BEX", "OE_PUS", "OE_PAR", "OE_MOB", "OE_IN_FIS", "OE_EX_FIS"]
    data_x = data_x.drop(columns=except_list_2)

    return data_x, data_y


def MICE(df):      

       ## encoding
       cati_list = ['SM' , 'DR']      

       data_encoded = pd.get_dummies(df , columns=cati_list , dtype=int)
       
       data_encoded['SM'] = data_x['SM']
       data_encoded['DR'] = data_x['DR']
       column_encoded = data_encoded.columns

    data_encoded.loc[data_encoded["SM"].isna(), "SM_1.0"] = np.nan
    data_encoded.loc[data_encoded["SM"].isna(), "SM_2.0"] = np.nan
    data_encoded.loc[data_encoded["SM"].isna(), "SM_3.0"] = np.nan
    data_encoded.loc[data_encoded["SM"].isna(), "SM_4.0"] = np.nan

    data_encoded.loc[data_encoded["DR"].isna(), "DR_0.0"] = np.nan
    data_encoded.loc[data_encoded["DR"].isna(), "DR_1.0"] = np.nan
    data_encoded.loc[data_encoded["DR"].isna(), "DR_2.0"] = np.nan

    ## impute ##
    imputer = IterativeImputer(max_iter=50, random_state=0)  ## enough max_iter
    imputer.fit(data_encoded)
    im_test = imputer.transform(data_encoded)

    test_last = pd.DataFrame(im_test, columns=column_encoded)
    yeah = ["PMH_MM", "SM_1.0", "SM_2.0", "SM_3.0", "SM_4.0", "DR_0.0", "DR_1.0", "DR_2.0"]
    test_last[yeah] = test_last[yeah].round(0)

    ## calculate BMI_R by BMI instead of MICE ##
    test_last.loc[(test_last["BMI"] < 18.5), "BMI_R"] = 0
    test_last.loc[(test_last["BMI"] >= 18.5) & (test_last["BMI"] < 23), "BMI_R"] = 1
    test_last.loc[(test_last["BMI"] >= 23) & (test_last["BMI"] < 25), "BMI_R"] = 2
    test_last.loc[(test_last["BMI"] >= 25) & (test_last["BMI"] < 30), "BMI_R"] = 3
    test_last.loc[(test_last["BMI"] >= 30), "BMI_R"] = 4



       ## decoding
       test_last.loc[(test_last['SM_1.0'] ==  1) , 'SM'] = 1
       test_last.loc[(test_last['SM_2.0'] ==  1) , 'SM'] = 2
       test_last.loc[(test_last['SM_3.0'] ==  1) , 'SM'] = 3
       test_last.loc[(test_last['SM_4.0'] ==  1) , 'SM'] = 4

    test_last.loc[(test_last["DR_0.0"] == 1), "DR"] = 0
    test_last.loc[(test_last["DR_1.0"] == 1), "DR"] = 1
    test_last.loc[(test_last["DR_2.0"] == 1), "DR"] = 2

    # drop
    yeah = ["SM_1.0", "SM_2.0", "SM_3.0", "SM_4.0", "DR_0.0", "DR_1.0", "DR_2.0"]
    test_last = test_last.drop(columns=yeah)
       test_last.loc[(test_last['DR_0.0'] ==  1) , 'DR'] = 0
       test_last.loc[(test_last['DR_1.0'] ==  1) , 'DR'] = 1
       test_last.loc[(test_last['DR_2.0'] ==  1) , 'DR'] = 2

       # drop
       yeah = ['SM_1.0', 'SM_2.0', 'SM_3.0', 'SM_4.0', 'DR_0.0', 'DR_1.0', 'DR_2.0']
       test_last = test_last.drop(columns=yeah)

    return test_last


def merge_f(data_x):
    ###### Dental risk factors ######
    # none==0 , IMP==1 , EXT==2 , DEN==3 , TOR==4 , RCT==5 , PE==6 , SPO==7
    or_list = ["OR_IMP", "OR_DEN", "OR_TOR", "OR_RCT", "OR_PE", "OR_SPO"]
    data_x["count"] = (
        data_x["OR_IMP"]
        + data_x["OR_EXT"]
        + data_x["OR_DEN"]
        + data_x["OR_TOR"]
        + data_x["OR_RCT"]
        + data_x["OR_PE"]
        + data_x["OR_SPO"]
    )
    data_x.loc[(data_x["count"] > 1), "OR_EXT"] = 1
    data_x.loc[(data_x["count"] > 1), or_list] = 0
    data_x = data_x.drop(columns="count")

    data_x["OR"] = 0

    data_x.loc[(data_x["OR_IMP"] == 1), "OR"] = 1
    data_x.loc[(data_x["OR_EXT"] == 1), "OR"] = 2
    data_x.loc[(data_x["OR_DEN"] == 1), "OR"] = 3
    data_x.loc[(data_x["OR_TOR"] == 1), "OR"] = 4
    data_x.loc[(data_x["OR_RCT"] == 1), "OR"] = 5
    data_x.loc[(data_x["OR_PE"] == 1), "OR"] = 6
    data_x.loc[(data_x["OR_SPO"] == 1), "OR"] = 7

    data_x = data_x.drop(columns=or_list)
    data_x = data_x.drop(columns="OR_EXT")

    ###### Low dosing / High dosing ######
    data_x["PMH_L_H"] = 0
    data_x.loc[(data_x["PMH_H"] == 1), "PMH_L_H"] = 1
    data_x = data_x.drop(columns="PMH_L")
    data_x = data_x.drop(columns="PMH_H")

    ###### Route of administration ######
    ra_list = ["PMH_PO", "PMH_IVSC", "PMH_CK"]
    data_x["PMH_RA"] = 0  # route of administration
    data_x.loc[(data_x["PMH_PO"] == 1), "PMH_RA"] = 1
    data_x.loc[(data_x["PMH_CK"] == 1), "PMH_RA"] = 3
    data_x.loc[(data_x["PMH_IVSC"] == 1), "PMH_RA"] = 2
    data_x = data_x.drop(columns=ra_list)

    ###### Denosumab ######
    data_x["PMH_t_deno"] = 0
    data_x.loc[(data_x["PMH_t_PRO"] == 1), "PMH_t_deno"] = 1
    data_x.loc[(data_x["PMH_t_XG"] == 1), "PMH_t_deno"] = 2
    data_x = data_x.drop(columns="PMH_t_PRO")
    data_x = data_x.drop(columns="PMH_t_XG")

    """
       ###### Bisphosphonate ######  only for Model2
       bi_list = ['PMH_t_ALEN' , 'PMH_t_BAN' , 'PMH_t_ZOLE' , 'PMH_t_RISED']
       data_x['PMH_t_bisph'] = 0
       data_x.loc[(data_x['PMH_t_ALEN'] ==  1) , 'PMH_t_bisph'] = 1
       data_x.loc[(data_x['PMH_t_BAN'] ==  1) , 'PMH_t_bisph'] = 2
       data_x.loc[(data_x['PMH_t_ZOLE'] ==  1) , 'PMH_t_bisph'] = 3
       data_x.loc[(data_x['PMH_t_RISED'] ==  1) , 'PMH_t_bisph'] = 4
       data_x = data_x.drop(columns=bi_list)
       """

    ###### Blood Pressure ######
    data_x["BP"] = (data_x["SBP"] + data_x["DBP"]) / 2
    data_x = data_x.drop(columns="SBP")
    data_x = data_x.drop(columns="DBP")

    return data_x


####path = 'D:/노트북/Work/보건복지부과제/ONJ/onj/clinical'

path = "/mnt/aix22301/onj/code/clinical"
os.chdir(path)
df = pd.read_excel("이대목동+이대서울_v1.1.xlsx")


### Preprocess ###
df = start(df)
[data_x, data_y] = data_make(df)
data_x = MICE(data_x)
#data_x.to_csv(path + '/X_EW_mice_2.csv')



#################
# nope==0 , IMP==1 , EXT==2 , DEN==3 , TOR==4 , RCT==5 , PE==6 , SPO==7
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
       data_x = data_x.drop(columns='OR_EXT')



       ###### Low dosing / High dosing ######
       data_x['PMH_L_H'] = 0
       data_x.loc[(data_x['PMH_H'] == 1) , 'PMH_L_H'] = 1
       data_x = data_x.drop(columns='PMH_L')
       data_x = data_x.drop(columns='PMH_H')

  

       ###### Route of administration ######
       ra_list = ['PMH_PO' , 'PMH_IVSC' , 'PMH_CK']
       data_x['PMH_RA'] = 0        # route of administration
       data_x.loc[(data_x['PMH_PO'] ==  1) , 'PMH_RA'] = 1
       data_x.loc[(data_x['PMH_CK'] ==  1) , 'PMH_RA'] = 3
       data_x.loc[(data_x['PMH_IVSC'] ==  1) , 'PMH_RA'] = 2          
       data_x = data_x.drop(columns=ra_list)



       ###### Denosumab ######
       data_x['PMH_t_deno'] = 0
       data_x.loc[(data_x['PMH_t_PRO'] ==  1) , 'PMH_t_deno'] = 1
       data_x.loc[(data_x['PMH_t_XG'] ==  1) , 'PMH_t_deno'] = 2
       data_x = data_x.drop(columns='PMH_t_PRO')
       data_x = data_x.drop(columns='PMH_t_XG')


       '''
       ###### Bisphosphonate ######  only for Model2
       bi_list = ['PMH_t_ALEN' , 'PMH_t_BAN' , 'PMH_t_ZOLE' , 'PMH_t_RISED']
       data_x['PMH_t_bisph'] = 0
       data_x.loc[(data_x['PMH_t_ALEN'] ==  1) , 'PMH_t_bisph'] = 1
       data_x.loc[(data_x['PMH_t_BAN'] ==  1) , 'PMH_t_bisph'] = 2
       data_x.loc[(data_x['PMH_t_ZOLE'] ==  1) , 'PMH_t_bisph'] = 3
       data_x.loc[(data_x['PMH_t_RISED'] ==  1) , 'PMH_t_bisph'] = 4
       data_x = data_x.drop(columns=bi_list)
       '''

  
       ###### Blood Pressure ######
       data_x['BP'] = (data_x['SBP'] + data_x['DBP'])/2
       data_x = data_x.drop(columns='SBP')
       data_x = data_x.drop(columns='DBP')



       return data_x




####path = 'D:/노트북/Work/보건복지부과제/ONJ/onj/clinical'
os.chdir(path)
df = pd.read_excel('이대목동+이대서울_v1.1.xlsx')


### Preprocess ###
df = start(df)
[data_x , data_y] = data_make(df)
data_x = MICE(data_x)
data_x = merge_f(data_x)

data_x.to_csv(path + '/data_X.csv') 
data_y.to_csv(path + '/data_Y.csv')





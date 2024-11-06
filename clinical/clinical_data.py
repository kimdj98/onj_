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


column_list = [
    "pt_H",
    "ONJ_DIA_AGE",
    "SEX",
    "HIG",
    "WEI",
    "BMI",
    "BMI_R",
    "SBP",
    "DBP",
    "LQ",
    "SM",
    "DR",
    "PM",
    "ONJ_DIA_CK",
    "ONJ_DIA_ST",
    "ONJ_DIA_DD",
    "ONJ_CP_DD",
    "CON_CP_DD",
    "SUR_DD_1",
    "SUR_DD_2",
    "SUR_DD_3",
    "CP_DD_un",
    "CP",
    "ONJ_OST",
    "ONJ_CC",
    "MH_HTN",
    "MH_DM",
    "MH_CVA",
    "MH_OA",
    "MH_RART",
    "MH_RF",
    "MH_HYP",
    "MH_HYPO",
    "MH_HYPE",
    "MH_CC",
    "PMH_L",
    "PMH_H",
    "PMH_BD",
    "PMH_DB",
    "PMH_PO",
    "PMH_IVSC",
    "PMH_CK",
    "PMH_MM",
    "PMH_MM_un",
    "PMH_t_PRO",
    "PMH_t_XG",
    "PMH_t_ALEN",
    "PMH_t_BAN",
    "PMH_t_ZOLE",
    "PMH_t_RISED",
    "PMH_t_STE",
    "PMH_t_RHEUM",
    "PMH_t_ROMO",
    "PMH_t_SERM",
    "PMH_C_ANTI",
    "PMH__ANT",
    "PMH_un",
    "OR_IMP",
    "OR_EXT",
    "OR_DEN",
    "OR_TOR",
    "OR_RCT",
    "OR_PE",
    "OR_SPO",
    "ONJ_SI_UL",
    "ONJ_SI_UC",
    "ONJ_SI_UR",
    "ONJ_SI_LL",
    "ONJ_SI_LC",
    "ONJ_SI_LR",
    "OE_PA",
    "OE_SW",
    "OE_BEX",
    "OE_PUS",
    "OE_PAR",
    "OE_MOB",
    "OE_IN_FIS",
    "OE_EX_FIS",
    "SUR_DC_MM",
    "SUR_RE_DC_MM",
    "SUR_DC_CON",
    "SUR_AN",
    "SUR_BMP",
    "SUR_PRF",
    "SU_PTH",
    "SUR",
    "SUR_RE",
]
# 'pt_H_ID', 'pt_CODE', 'PMH_OTHERS', 'Early Wide resection', 'PTH_Wide resection', 'PTH_Sequestration', 'Sequestration_Only', 'Conservative',

except_list = [
    "pt_H",
    "ONJ_DIA_CK",
    "ONJ_DIA_ST",
    "ONJ_DIA_DD",
    "ONJ_CP_DD",
    "CON_CP_DD",
    "SUR_DD_1",
    "SUR_DD_2",
    "SUR_DD_3",
    "CP_DD_un",
    "CP",
    "ONJ_SI_UL",
    "ONJ_SI_UC",
    "ONJ_SI_UR",
    "ONJ_SI_LL",
    "ONJ_SI_LC",
    "ONJ_SI_LR",
    "SUR_DC_MM",
    "SUR_RE_DC_MM",
    "SUR_DC_CON",
    "SUR_AN",
    "SUR_BMP",
    "SUR_PRF",
    "SU_PTH",
    "SUR",
    "SUR_RE",
]
# 'pt_H_ID', 'pt_CODE' , 'Early Wide resection', 'PTH_Wide resection', 'PTH_Sequestration', 'Sequestration_Only', 'Conservative',


def onj_convert(data):
    data["true"] = 0
    data["true"] = np.where(data["ONJ_DIA_CK"] != "O", 0, 1)
    data["ONJ_DIA_CK"] = data["true"]
    data = data.drop(columns="true")

    return data


def data_make(data):

    # data = shuffle(data)
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

    ###### data_y ######
    data_y = data["ONJ_DIA_CK"]
    data_y = data_y.astype(dtype=np.float64)

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
    cati_list = ["SM", "DR"]

    data_encoded = pd.get_dummies(df, columns=cati_list, dtype=int)

    data_encoded["SM"] = data_x["SM"]
    data_encoded["DR"] = data_x["DR"]
    column_encoded = data_encoded.columns

    data_encoded.loc[data_encoded["SM"].isna(), "SM_1.0"] = np.nan
    data_encoded.loc[data_encoded["SM"].isna(), "SM_2.0"] = np.nan
    data_encoded.loc[data_encoded["SM"].isna(), "SM_3.0"] = np.nan
    data_encoded.loc[data_encoded["SM"].isna(), "SM_4.0"] = np.nan

    data_encoded.loc[data_encoded["DR"].isna(), "DR_0.0"] = np.nan
    data_encoded.loc[data_encoded["DR"].isna(), "DR_1.0"] = np.nan
    data_encoded.loc[data_encoded["DR"].isna(), "DR_2.0"] = np.nan

    ## mice
    imputer = IterativeImputer(max_iter=50, random_state=0)
    imputer.fit(data_encoded)
    im_test = imputer.transform(data_encoded)

    ## df, round
    test_last = pd.DataFrame(im_test, columns=column_encoded)
    yeah = ["PMH_MM", "PMH_MM_un", "SM_1.0", "SM_2.0", "SM_3.0", "SM_4.0", "DR_0.0", "DR_1.0", "DR_2.0"]
    test_last[yeah] = test_last[yeah].round(0)

    ## BMI_R calculation
    test_last.loc[(test_last["BMI"] < 18.5), "BMI_R"] = 0
    test_last.loc[(test_last["BMI"] >= 18.5) & (test_last["BMI"] < 23), "BMI_R"] = 1
    test_last.loc[(test_last["BMI"] >= 23) & (test_last["BMI"] < 25), "BMI_R"] = 2
    test_last.loc[(test_last["BMI"] >= 25) & (test_last["BMI"] < 30), "BMI_R"] = 3
    test_last.loc[(test_last["BMI"] >= 30), "BMI_R"] = 4

    ## manually decoding..
    test_last.loc[(test_last["SM_1.0"] == 1), "SM"] = 1
    test_last.loc[(test_last["SM_2.0"] == 1), "SM"] = 2
    test_last.loc[(test_last["SM_3.0"] == 1), "SM"] = 3
    test_last.loc[(test_last["SM_4.0"] == 1), "SM"] = 4

    test_last.loc[(test_last["DR_0.0"] == 1), "DR"] = 0
    test_last.loc[(test_last["DR_1.0"] == 1), "DR"] = 1
    test_last.loc[(test_last["DR_2.0"] == 1), "DR"] = 2

    # drop
    yeah = ["SM_1.0", "SM_2.0", "SM_3.0", "SM_4.0", "DR_0.0", "DR_1.0", "DR_2.0"]
    test_last = test_last.drop(columns=yeah)

    return test_last


path = "/mnt/aix22301/onj/code/clinical/clinical"
os.chdir(path)

df = pd.read_excel("이대목동+이대서울_v1.1.xlsx", names=column_list)
df.to_csv("ONJ_patient_clinical.csv")
df = pd.read_csv("ONJ_patient_clinical.csv", index_col=0)


[data_x, data_y] = data_make(df)
data_x = MICE(data_x)
# data_x.to_csv(path + '/X_EW_mice_2.csv')


#################
# nope==0 , IMP==1 , EXT==2 , DEN==3 , TOR==4 , RCT==5 , PE==6 , SPO==7
or_list = ["OR_IMP", "OR_EXT", "OR_DEN", "OR_TOR", "OR_RCT", "OR_PE"]
data_x["count"] = (
    data_x["OR_IMP"] + data_x["OR_EXT"] + data_x["OR_DEN"] + data_x["OR_TOR"] + data_x["OR_RCT"] + data_x["OR_PE"]
)
data_x.loc[(data_x["count"] > 1), "OR_SPO"] = 1
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
data_x = data_x.drop(columns="OR_SPO")


### save ###
## final in : '이대목동+이대서울_241002.xlsx' >> out : X_EW.csv, Y_EW.csv
data_x.to_csv(path + "/X_EW.csv")
data_y.to_csv(path + "/Y_EW.csv")

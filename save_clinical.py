import pandas as pd
import os

# 'ONJ_AI임상정보.xlsx' is the original excel file from google drive
# 'ONJ_AI_clinical_modified.xlsx' is the column-modified file of 'ONJ_AI임상정보.xlsx'
# 'ONJ_patient_clinical.csv' is the 'ONJ_AI_clinical_modified.xlsx' converted to csv

# Original excel file from google drive : 'ONJ_AI임상정보.xlsx'
# Column-modified file of the original : 'ONJ_clinical_modified.xlsx'
# 


#local_path = '/home/sesameOil/Downloads'
path = '/mnt/4TB1/onj/dataset/v0'


'''
## save excel file to onj/dataset/v0
excel1 = pd.read_excel(local_path + '/ONJ_AI임상정보.xlsx')
excel2 = pd.read_excel(local_path + '/ONJ_AI_clinical_modified.xlsx')
excel1.to_excel(path + '/ONJ_AI임상정보.xlsx')
excel2.to_excel(path + '/ONJ_AI_clinical_modified.xlsx')
'''

## convert to csv and read
data = pd.read_excel(path + '/ONJ_AI_clinical_modified.xlsx')
data.to_csv(path + '/ONJ_patient_clinical.csv')                         
data = pd.read_csv(path + '/ONJ_patient_clinical.csv')





data = data.drop(columns='Unnamed: 0')


column_list = ['Registration Number', 'Code Number', 'AGE', 'SEX', 'HIG',
       'WEI', 'BMI', 'BMI_R', 'SBP', 'DBP', 'LQ', 'SM', 'DR', 'PM', 'ONJ',
       'Stage', 'Diagnosis Date', 'ONJ Cured Date', 'Completion Date',
       'Surgery Date1', 'Surgery Date2', 'Surgery Date3', 'Osteoporosis',
       'Cancer', 'Hypertension', 'Diabetes Mellitus', 'CVA', 'Osteoarthritis',
       'Rheumatic Arthritis', 'Renal Failure', 'Hyperlipidemia',
       'Hypothyroidism', 'Hyperthyroidism', 'Low dosing', 'High dosing',
       'Combination (BP-Deno)', 'Combination (Deno-BP)', 'PO', 'IV',
       'Dont Know', 'Month of Taking', 'Prolia', 'Xgeva', 'Alendronate',
       'Ibandronate', 'Zoledronate', 'Risedronate', 'Steroid',
       'Rheumatism Drug', 'Romosozumab', 'Etc', 'Unknown Antiresorptives',
       'Implant Associated Case', 'Extraction', 'Denture', 'Torus',
       'Periapical Lesion', 'Periodontitis', 'Spontaneous', 'UL', 'UC', 'UR',
       'LL', 'LC', 'LR', 'Pain', 'Swelling', 'Exposure', 'Abscess',
       'Paresthesia', 'Mobility', 'Oroantral Fistula', 'Orocutaneous Fistula',
       'Month of Drug Cessation', 'Month of Drug Holiday', 'Anesthesia', 'BMP',
       'PRF', 'PTH', 'Early Wide resection', 'PTH_Wide resection',
       'PTH_Sequestration', 'Sequestration_Only', 'Conservative',
       'ONJ Relapse']






for i in range(len(data)):
    profile = data.iloc[i]
    patient_code = profile['Code Number']
    

    if profile['ONJ']==1 :       # if ONJ 
              
        if not os.path.exists(path + '/ONJ_labeling/' + str(patient_code)):
            print(str(patient_code)) 
        #    os.mkdir(path + '/Non-ONJ/' + str(patient_code))
                       
        else :
            profile = profile.to_frame().transpose()[column_list].reset_index(drop=True)
            profile.to_csv(path + '/ONJ_labeling/' + str(patient_code) + '/Clinical_info.csv')

        



    elif profile['ONJ']==0 :     # if Non-ONJ
        
        if not os.path.exists(path + '/Non_ONJ_soi/' + str(patient_code)):
            print(str(patient_code))
        #    os.mkdir(path + '/ONJ/' + str(patient_code))
            
        else :
            profile = profile.to_frame().transpose()[column_list].reset_index(drop=True)
            profile.to_csv(path + '/Non_ONJ_soi/' + str(patient_code) + '/Clinical_info.csv')

        


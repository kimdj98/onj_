** Input Data : 이대목동+이대서울_v1.1.xlsx

[ Code Explanation ]
>> clinical_data : data preprocessing code
>> clinical_gridSearch_model : code including both hyperparameter tuning and model making. GridSearch is used.
>> clinical_gridSearch_shap : SHAP code for the model made by GridSearch
>> clinical_hyperopt_parTune : Tuning hyperparameters by Hyperopt (Bayesian Optimization)
>> clinical_hyperopt_makeModel : Making model with the parameters tuned from '~parTune.py'
>> clinical_hyperopt_shap : SHAP code for the model made by Hyperopt




[ Manual~★ ]
0. Check the path and file names, model names for all codes.


1. GridSearch Model
	1-1. clinical_gridSearch_model.py
	(1) check path and file names of input data (L276~278)
	(2) just run the whole file
	(3) def 'find_par'==tuning, 'make_model'==make model


2. Hyperopt Model
	2-1. clinical_hyperopt_parTune.py
	(1) check path and file names of input data (L125~127)
	(2) run
	(3) results are printed
		EX)) best:  {'batch_size': 55.50482816274457, 'dropout1': 0.424514199140625, 
			  'dropout2': 0.4531309216221532, 'num_layers': 0, 'optimizer': 1, 
			  'units1': 169.25139118961928, 'units2': 936.8870653560863}

		>>>> this means :  batch_size : 56 (rounded)
				dropout1 : 0.425 (rounded)
				dropout2 : 0.453 (rounded)
				num_layers : two (index of list)
				optimizer : adam (index of list)
				units1 : 169 (rounded)
				units2 : 937 (rounded)


	2-2. clinical_hyperopt_makeModel.py
	(1) check
	(2) set the hyperparameters (L17~23)(num of layers!!)
	(3) check if the right optimizer is selected (L77~79) 
	(4) run


	2-3. clinical_hyperopt_shap.py
	(1) check
	(2) set the hyperparameters (L26~32)
	(3) run



++ the data is split each time the code is run, but since the seed is fixed no problem



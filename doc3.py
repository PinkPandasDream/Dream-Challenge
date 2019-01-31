# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import math
from scipy import stats
from sklearn import preprocessing


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE


dataset_training = pd.read_csv("training_df_features.csv")

" ---------- PRE PROCESSAMENTO ---------- "


"---------- Eliminar valores de kd=0 ou sem valor de kd ----------"

##valores do KD estao na coluna standard_value


print("Dimensao inicial do dataset (linhas, colunas):", dataset_training.shape)

#Obter uma lista com o nome das colunas do dataset
#print(list(dataset.columns.values))

#linhas a eliminar porque tem kd=0 ou kd=nan

del_rows = []

del_rows = dataset_training.loc[dataset_training["standard_value"] == 0].index

for x in range(dataset_training.shape[0]):
    if math.isnan(dataset_training.loc[x,"standard_value"]) == True:
        del_rows.append(x)
        
#ordenar as linhas a eliminar
del_rows_sorted = sorted(del_rows)
#print(del_rows_sorted)

#eleminar as linhas necessarias
clean_dataset = dataset_training.drop(del_rows_sorted).reset_index(drop=True)

print("Dimensao do dataset depois de eliminar as linhas com kd=0 ou kd=nan (linhas, colunas):", clean_dataset.shape)


"---------- Eliminar valores omissos ----------"

print(clean_dataset.isnull().values.any())
print(clean_dataset.shape)


"---------- Verificar a normalidade ----------"

#selecionar colunas com as features
dataset = clean_dataset.iloc[:,25:]
#print(dataset)


#verificar normalidade
normality_training = stats.normaltest(dataset.all())
alpha = 1e-3 
if normality_training[1]<alpha:
    print("Does not follow a normal distribution")
else:
    print("Follows a normal distribution")
    
    
#normalizar
train_norm = preprocessing.normalize(dataset, norm="l2")
normalizer_train = preprocessing.Normalizer().fit(dataset)
train_n = normalizer_train.transform(dataset)
train = pd.DataFrame(train_n, columns = dataset.columns, index = dataset.index)
#print(train.head)






"---------- FEATURE SELECTION ----------"

selection_data = train

del_columns = [] #colunas a eliminar serao colocadas aqui



###############################################################

" ---------- Filter Methods: Variance Threshold ---------- "

var_thresh = VarianceThreshold(0.0)
var_thresh.fit(selection_data)
selected = var_thresh.get_support()

temp = np.where(selected == False)
for x in temp[0]:
    del_columns.append(x)
#print(del_columns)

#ordenar as colunas a eliminar
del_columns_sorted = sorted(del_columns)
#print(del_columns_sorted)

#eliminar as colunas com variancia = 0
clean_dataset_temp = selection_data.drop(selection_data.columns[del_columns_sorted], axis = 1).reset_index(drop=True)
print("Dimensao do selection_data depois de eliminar as col com var = 0 (linhas, colunas):", clean_dataset_temp.shape)
del_columns = []
del_columns_sorted = []
#print(del_columns)

target_data = clean_dataset["standard_value"]


###############################################################

" ---------- Wrapper methods: RFE - Recursive feature elimination ---------- "

estimator = SVR(kernel = "linear")
select_rfe = RFE(estimator,n_features_to_select = None, step = 1)
select_rfe.fit(clean_dataset_temp, target_data)
selected_columns = select_rfe.support_


temp1 = np.where(selected_columns == False)
print("Nº de colunas analisadas pelo RFE:",selected_columns.shape)
for x in temp1[0]:
    del_columns.append(x)

print("Nº de colunas a eliminar:",len(del_columns))
    
#ordenar as colunas a eliminar
del_columns_sorted = sorted(del_columns)
#print(del_columns_sorted)
    
clean_dataset_final = clean_dataset_temp.drop(clean_dataset_temp.columns[del_columns_sorted], axis = 1).reset_index(drop=True)
print("Dimensao do clean_dataset_final (linhas, colunas):", clean_dataset_final.shape)

#juntar novamente as colunas nao numericas (indices 0:24) ao dataset
dataset_final = pd.concat([clean_dataset.iloc[:,:25], clean_dataset_final], axis = 1)
print("Dimensao do dataset final (linhas, colunas):", dataset_final.shape)

#guardar o dataset "limpo" num ficheiro csv
dataset_final.to_csv("final_dataset.csv")



# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np

from scipy import stats
from sklearn import preprocessing

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE


dataset_training = pd.read_csv(r"training_df.csv")

print(dataset_training)

"---------- ANÁLISE EXPLORATÓRIA ----------"
print(dataset_training.shape)


"---------- EXTRAÇÃO DE FEATURES ----------"

import numpy as np
import pandas as pd
from pydpi import pydpi
from pydpi import pydrug
from pydpi.pydrug import getmol,Chem, kappa
from pydpi.drug import fingerprint,getmol
from rdkit.Chem.AtomPairs import Pairs

from itertools import chain
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate

column_smiles = ['smiles']
path="C:\\Users\\Acer\\Desktop\\uminho\\2ano\\SIB\\dream\\nosso\\training_df_sara.csv" # dataset path
export_path_morgan="C:\\Users\\Acer\\Desktop\\uminho\\2ano\\SIB\\dream\\nosso\\morgan.csv"
export_path_mollog="C:\\Users\\Acer\\Desktop\\uminho\\2ano\\SIB\\dream\\nosso\\mollog.csv"
export_path_maccs="C:\\Users\\Acer\\Desktop\\uminho\\2ano\\SIB\\dream\\nosso\\maccs.csv"
export_path_interaction="C:\\Users\\Acer\\Desktop\\uminho\\2ano\\SIB\\dream\\nosso\\features_interact.csv"


##############################################################################
                      # FEATURES INTERACAO PROTEINA-MOLECULA
#############################################################################


                  # result folder
dpi = pydpi.PyDPI()                                                 # variavel para o script da interacao

column_names = ['smiles', 'target_id']                              # list de colunas a extrair

result_list_F1 = []                                                 # lista de resultados (F1 feature)
result_list_F2 = []                                                 # lista de resultados (F2 feature)
target_id_errors = []                                               # lista de erros

def get_f1_f2(row):
    try:
        protein_sequence = dpi.GetProteinSequenceFromID(row['target_id']) 
        dpi.ReadProteinSequence(protein_sequence)
        aa_composition = dpi.GetAAComp() #COMPOSICAO AMINOACIOS
        molecule = Chem.MolFromSmiles(row['smiles']) 
        kappa_descriptors = kappa.GetKappa(molecule)

        if(row.name % 500 == 0):                                                # para facilitar o processo, a leitura e feita aos poucos
            partial = pd.DataFrame(result_list_F1)                              # os smiles e target_id das colunas que dao erros que sao guardados num ficheiro
            partial.to_csv(path + "export_partial_f1.csv")               
            partial = pd.DataFrame(result_list_F2)
            partial.to_csv(path + "export_partial_f2.csv")
            partial = pd.DataFrame(target_id_errors)
            partial.to_csv(path + "errors.csv")
        
        result_list_F1.append(dpi.GetDPIFeature1(kappa_descriptors, aa_composition))
        result_list_F2.append(dpi.GetDPIFeature2(kappa_descriptors, aa_composition))
    except:
        dic = {'smiles':row['smiles'], 'target_id':row['target_id']}
        target_id_errors.append(dic)

    print(row.name)


df = pd.read_csv(path, header=0, skipinitialspace=True, usecols=column_names)#, nrows=15)

df.apply(get_f1_f2, axis=1)                                                         

result_dataframe = pd.DataFrame(result_list_F1)
result_dataframe.to_csv(export_path_interaction + "export_total_f1.csv")                # csv das features f1 do pydpi das linhas que restam 
result_dataframe = pd.DataFrame(result_list_F2)
result_dataframe.to_csv(export_path_interaction + "export_total_f2.csv")                # csv das features f2 do pydpi das linhas que restam 

target_id_errors_dataframe = pd.DataFrame(target_id_errors)
target_id_errors_dataframe.to_csv(export_path_interaction + "errors.csv")





##############################################################################
                     # FEATURES COMPOSTOS MOLECULARES
##############################################################################



column_smiles = ['smiles']              #lista de colunas a extrair para estas featutes
############################################## 2DFingerprint #muito pesado (acabou por não se usar)
'''
def _2DFingerprint(molecule):
    desc = Generate.Gen2DFingerprint(Chem.MolFromSmiles(molecule), Gobbi_Pharm2D.factory)
    arr = np.array(desc)

    return arr


dff = pd.read_csv(path, header=0, skipinitialspace=True, usecols=column_names)#, nrows=15)          
dff['2Dfingerprint'] = dff['smiles'].apply(lambda x: _2DFingerprint(x))

list_size = len(dff['2Dfingerprint'][0])
prefix = "2D_"                                                                              
list_of_headers = [prefix + str(i+1) for i in range(list_size)]                                     #headers para cada bit de 2dfingerprint

list_of_dicts = [] 
for row in dff['2Dfingerprint']:
    list_of_dicts.append(dict(zip(list_of_headers, row)))                                           # criacao de uma lista de dicionarios com os headers e as linhas (que contêm listas de bits)
                                                                                                    # o mesmo foi aplicado para os morgan e maccs

dff = dff.merge(pd.DataFrame(list_of_dicts), left_index=True, right_index=True)

'''

###############################################################

" ---------- Morgan Fingerprint ---------- "

def Morgan_vect(smiles):
    mol=Chem.MolFromSmiles(smiles)
    vector = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024)
    arr = np.array(vector)
    return arr

dff = pd.read_csv(path, header=0, skipinitialspace=True, nrows=10, usecols=column_smiles)
dff['Morgan'] = dff['smiles'].apply(lambda x: Morgan_vect(x))

list_size = len(dff['Morgan'][0])
prefix = "Morgan_"
list_of_headers = [prefix + str(i+1) for i in range(list_size)]

list_of_dicts = [] 
for row in dff['Morgan']:
    list_of_dicts.append(dict(zip(list_of_headers, row)))


df_Morgan = dff.merge(pd.DataFrame(list_of_dicts), left_index=True, right_index=True)
#dff.to_csv(export_path_morgan)
#print 'done'

############################################################### 

" ---------- Maccs ---------- "

def maccs_keys(smiles):
    # mol=Chem.MolFromSmiles(row['smiles']) #aqui entra os smiles
    # res=fingerprint.CalculateMACCSFingerprint(mol)    isto seria se nao fosse vetor
    # result_maccs.append(res)
    mol=Chem.MolFromSmiles(smiles)
    fps=rdMolDescriptors.GetMACCSKeysFingerprint(mol)
	# DataStructs.ConvertToNumpyArray(desc, arr)
    arr = np.array(fps)
    return arr

dff = pd.read_csv(path, header=0, skipinitialspace=True, nrows=10, usecols=column_smiles)
dff['MACCS'] = dff['smiles'].apply(lambda x: maccs_keys(x))

list_size = len(dff['MACCS'][0])
prefix = "MACCS_"
list_of_headers = [prefix + str(i+1) for i in range(list_size)]

list_of_dicts = [] 
for row in dff['MACCS']:
    list_of_dicts.append(dict(zip(list_of_headers, row)))


df_MACCS = dff.merge(pd.DataFrame(list_of_dicts), left_index=True, right_index=True)
#dff.to_csv(export_path_maccs)
#print 'done'
#print(dff)

###############################################################

" ---------- MolLogP ---------- "

def getMolLogP(smile):
    ms=Chem.MolFromSmiles(smile)
    descrit= Descriptors.MolLogP(ms)            #descritores MolLogP
    return descrit

dff = pd.read_csv(path, header=0, skipinitialspace=True, nrows=10, usecols=column_smiles)
MolLog_df = dff['smiles'].apply(lambda x: getMolLogP(x))
MolLog_df.to_csv(export_path_mollog)

############################################################### AtomPairFingerprint Nao binario 2048
'''
atompair=[]
def AtomPairFingerprint(molecule_smile):
    #dic={}
    ms=Chem.MolFromSmiles(molecule_smile)
    desc = rdMolDescriptors.GetHashedAtomPairFingerprint(ms)
    #int(desc.GetLength())
    #for x in range(desc.GetLength()):
    #    dic['itens']=desc.__getitem__(x)
    #arr = np.array(desc) 
    for x in range(int(desc.GetLength())):
        atompair.append(desc.__getitem__(x))

'''
################################################################ concat

train=pd.read_csv(path)
train.drop(train.columns[0], axis=1, inplace=True)
final=pd.concat([train, MolLog_df,df_MACCS,df_Morgan], join='outer')


final.to_csv('features1_2.csv') 












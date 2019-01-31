import numpy as np
import pandas as pd
from itertools import chain
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
from rdkit.Chem.AtomPairs import Pairs
column_smiles = ['smiles']              #lista de colunas a extrair para estas featutes

##############################################
# FEATURES MOLECULAS
###############################################



############################################## 2DFingerprint #muito pesado

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



###############################################################     #Morgan


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

############################################################### Maccs
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

############################################################### MolLogP
def getMolLogP(smile):
    ms=Chem.MolFromSmiles(smile)
    descrit= Descriptors.MolLogP(ms)            #descritores MolLogP
    return descrit

dff = pd.read_csv(path, header=0, skipinitialspace=True, nrows=10, usecols=column_smiles)
MolLog_df = dff['smiles'].apply(lambda x: getMolLogP(x))
MolLog_df.to_csv(export_path_mollog)

############################################################### AtomPairFingerprint Nao binario 2048

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


################################################################ concat

train=pd.read_csv(path)
train.drop(train.columns[0], axis=1, inplace=True)
final=pd.concat([train, MolLog_df,df_MACCS,df_Morgan], join='outer')
#print final
#final.to_csv('features1_2.csv') 

import numpy as np
import pandas as pd
from pydpi import pydpi
from pydpi import pydrug
from pydpi.pydrug import getmol,Chem, kappa
from pydpi.drug import fingerprint,getmol
from rdkit.Chem.AtomPairs import Pairs

path="training_df.csv" # dataset path
export_path_interaction="features_interact.csv"

##############################################
# FEATURES INTERACOA PROTEINA-MOLECULA
###############################################  
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
            partial.to_csv(export_path + "export_partial_f1.csv")               
            partial = pd.DataFrame(result_list_F2)
            partial.to_csv(export_path + "export_partial_f2.csv")
            partial = pd.DataFrame(target_id_errors)
            partial.to_csv(export_path + "errors.csv")
        
        result_list_F1.append(dpi.GetDPIFeature1(kappa_descriptors, aa_composition))
        result_list_F2.append(dpi.GetDPIFeature2(kappa_descriptors, aa_composition))
    except:
        dic = {'smiles':row['smiles'], 'target_id':row['target_id']}
        target_id_errors.append(dic)

    print row.name


df = pd.read_csv(path, header=0, skipinitialspace=True, usecols=column_names)#, nrows=15)

df.apply(get_f1_f2, axis=1)                                                         

result_dataframe = pd.DataFrame(result_list_F1)
result_dataframe.to_csv(export_path_interaction + "export_total_f1.csv")                # csv das features f1 do pydpi das linhas que restam 
result_dataframe = pd.DataFrame(result_list_F2)
result_dataframe.to_csv(export_path_interaction + "export_total_f2.csv")                # csv das features f2 do pydpi das linhas que restam 

target_id_errors_dataframe = pd.DataFrame(target_id_errors)
target_id_errors_dataframe.to_csv(export_path_interaction + "errors.csv")


# -*- coding: utf-8 -*-


import pandas as pd
import urllib
import math
from pydpi import pypro
from pydpi.pypro import AAComposition
from pydpi.pypro import PyPro



"---------- EXTRAÇÃO DE FEATURES (CONTINUAÇÃO) ----------"



##############################################################################
                      # FEATURES PROTEINA
#############################################################################

origin=pd.read_csv('domain_sequences.csv')    #ficheiro contendo domínios das proteínas
origin.drop(origin.columns[0], axis=1, inplace=True)
target, ids=origin.iloc[:,0], origin.iloc[:,2]

dprot={}

for i in range(len(ids)):
	nome=target[i]
    ps=ids[i]
    protein=PyPro()
    protein.ReadProteinSequence(ps)
    
    temp=pypro.ProteinCheck(ps)
    aacomp=AAComposition.CalculateAAComposition(ps)   
    mor=protein.GetMoranAuto()
    moran=[mor[x] for x in mor]
    pseudo_aa=protein.GetPAAC(lamda=5, weight=0.05)
    dprot[i]=(nome,ps,aacomp,moran,pseudo_aa)
    
feat_prot=pd.DataFrame.from_dict(dprot, orient='index', columns=['target_id','ps','aacomp','moran','pseudo_aa'])    

train=pd.read_csv('features1_2.csv')
train.drop(train.columns[0], axis=1, inplace=True)

final=pd.concat([train, feat_prot], on='target_id', join='left')
final.to_csv('training_df_features.csv') 
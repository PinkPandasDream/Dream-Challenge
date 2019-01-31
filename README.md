# Dream-Challenge

No âmbito da unidade curricular de Sistemas Inteligentes para a Bioinformática foi nos proposto um trabalho de análise através do uso da linguagem de programação Python. IDG-DREAM Drug-Kinase Binding Prediction Challenge tem como objetivo avaliar o poder de modelos estatísticos e de aprendizagem máquina como meio sistemático de forma a direcionar os esforços de mapeamento de interação composto-alvo, priorizando interações mais potentes para posteriormente proceder à avaliação experimental. Este desafio está concentrado nos inibidores de quinase devido à sua importância clínica

Organização dos scripts:

Doc1.py contém: Análise Exploratória, Extração de features Interação Proteína-Molécula, Extração de features Compostos Moleculares (2D Fingerprint, Morgan Fingerprint, Maccs, MolLogP, Atom Pair Fingerprint)

Doc2.py contém: Extração de features Proteínas (domínio, composição em aminoácidos, Moran, pacc (Pseudo amino acid composition descriptors))

Doc3.py contém: Pré processamento (Eliminar valores de kd= 0 e valores de kd ausentes, Eliminar valores omissos, Verificar a normalidade dos dados, Normalização dos dados), Feature Selection (Variance Threshold, Wrapper methods: RPE - Recursive feature elimination)

Doc4.py contém: SVR Prediction (medidas de erro (MAE, MSE, RMSE), accuracy), Random Forest Prediction (medidas de erro (MAE, MSE, RMSE), accuracy), Linear Regression (medidas de erro (MAE, MSE, RMSE), accuracy)

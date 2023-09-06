# Write the script to preprocess data, train and evaluate your model here.

# bibliotecas utilizadas.

import pandas as pd
from random import randint
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import ExtraTreesClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix 
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV 
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

#importando os dados.

df_hindi = pd.read_csv('C:/Users/mathe/OneDrive/Área de Trabalho/magroove-ds-test/dataset-hindi.csv')
df_spanish = pd.read_csv('C:/Users/mathe/OneDrive/Área de Trabalho/magroove-ds-test/dataset-spanish.csv')

#Primeiro foi criado uma variável target para ter uma classificação do artista como indiano ou espanhol.
# 0 = hind.
# 1 = spanish.

i = 0
Target_h = []
for i in range(5000):
    numeros = randint(0,0)
    Target_h.append(numeros)
    print(numeros)

Target_h = pd.DataFrame(Target_h) #transformando a lista criada em dataframe.
df_hindi = pd.concat([df_hindi,Target_h],axis = 1) #junta um dataframe ao outro.
df_hindi.rename(columns={0: 'Target'}, inplace = True) #renomeando para ser mais fácil de identificar o target.


i = 1
Target_s = []
for i in range(5000):
    numeros = randint(1,1)
    Target_s.append(numeros)
    print(numeros)

Target_s = pd.DataFrame(Target_s) #transformando a lista criada em dataframe.
df_spanish = pd.concat([df_spanish,Target_s],axis = 1) #junta um dataframe ao outro.
df_spanish.rename(columns={0: 'Target'}, inplace = True) #renomeando para ser mais fácil de identificar o target.

#junta um dataframe ao outro.
df = pd.concat([df_hindi, df_spanish])

df.isnull().sum() # além de fazer o que a função acima faz, essa ainda soma os valores faltantes de cada coluna.

df.dropna(axis=0,inplace=True) #excluindo as colunas com dados faltantes. Optei por excluir as colunas com dados faltantes por serem poucas informações e além disso, não consegui imaginar uma associação matemática para relacionar os dados faltantes.

#%% one hot
#nesta parte foi realizado a troca das variáveis qualitavas por códigos binários para que o algoritmo consiga ler as informações. 

one_hot = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [0, 1, 2, 3])], remainder = 'passthrough')
df = one_hot.fit_transform(df).toarray()
df = pd.DataFrame(df)
df = df.drop(columns=[19033]) #excluindo uma das colunas de target para ter somente um target.
df.rename(columns={19034: 'Target'}, inplace = True)

#cirando duas variaveis para diminuir o tabalho do dataframe.
Target = df['Target']
df = df.drop(columns=['Target'])
#%% reduzindo o banco de dados para os dados ficarem mais fluidos quando eles forem usados em machine learning e fora isso para eles performarem melhor nos modelos.

pca = PCA(n_components = 7000)
df = pca.fit_transform(df)

# para ver o tanto de informação que se manteve do modelo anterior
df_2 = pca.explained_variance_ratio_ 
sum(df_2)*100

#detalhes finais antes do machine learning.
df= pd.DataFrame(df)

df = pd.concat([df, Target], axis = 1)


#%% machine learning

# em x estão todas as informações e em y estão as respostas para serem alcançadas
x = df.drop(columns=['Target'])
y = df['Target']

# melhores parametros DecisionTreeClassifier()
params = {'max_depth':[2,3,4,5,6,7,8],
          'criterion':['gini','entropy'],
          'class_weight':[None,'balanced'],
          'splitter': ['best', 'random']}

grid_search = GridSearchCV(estimator = DecisionTreeClassifier(),param_grid = params)

grid_search.fit(x,y)

melhores_parametros_DecisionTreeClassifier = grid_search.best_params_
melhor_resultado_DecisionTreeClassifier = grid_search.best_score_

# melhores parametros RandomForestClassifier

params_2 = {'criterion':['gini', 'entropy', 'log_loss'], 
            'max_features':['sqrt', 'log2', None]}

grid_search_2 = GridSearchCV(estimator = RandomForestClassifier(),param_grid = params_2)

grid_search_2.fit(x,y)

melhores_parametros_RandomForestClassifier = grid_search_2.best_params_
melhor_resultado_RandomForestClassifier = grid_search_2.best_score_

# melhores parametros ExtraTreesClassifier
params_3 = {'criterion':['gini', 'entropy', 'log_loss'],
            'max_features':['sqrt', 'log2', None]}

grid_search_3 = GridSearchCV(estimator = ExtraTreesClassifier(),param_grid = params_3)

grid_search_3.fit(x,y)

melhores_parametros_ExtraTreesClassifier = grid_search_3.best_params_
melhor_resultado_ExtraTreesClassifier = grid_search_3.best_score_

#separando os dados em modelos de teste e treino

[x_train, x_test, y_train, y_test] = train_test_split( x,y, test_size = 0.2, random_state = 0)

#árvore de decisão

modelo_1 = DecisionTreeClassifier(criterion = 'entropy', max_depth = 8, splitter = 'random' , class_weight = 'balanced', random_state = 0) #forma da qual estou usando a árvore de decisão.
modelo_1.fit(x_train,y_train) # treinando o modelo para gerar resultados.
modelo_1.feature_importances_ # mostra a importância de cada variável.

previsoes_1 = modelo_1.predict(x_test) # jogando os dados de teste para tentar prever.
acuracia_1 = accuracy_score(y_test, previsoes_1) # teste de acurácia para ver se o modelo foi eficiente.
Matriz_Confusao_1 = confusion_matrix (y_test,previsoes_1) # matriz de confusão para ver a quantidade de cada informação que ficou correta.
precision_1 = precision_score(y_test, previsoes_1, average = None) # teste da precisão do modelo.
recall_1 = recall_score(y_test, previsoes_1, average = None) # teste de sensibilidade do modelo.



# modelo de decisão rando forest.

random_forest_df = RandomForestClassifier(n_estimators = 10, criterion = 'gini', max_features = 'sqrt',random_state = 0)
random_forest_df.fit(x_train, y_train)

previsoes_2 = random_forest_df.predict(x_test) # jogando os dados de teste para tentar prever.
acuracia_2 = accuracy_score(y_test, previsoes_2) # teste de acurácia para ver se o modelo foi eficiente.
Matriz_Confusao_2 = confusion_matrix (y_test,previsoes_2) # matriz de confusão para ver a quantidade de cada informação que ficou correta.
precision_2 = precision_score(y_test, previsoes_2, average = None) # teste da precisão do modelo.
recall_2 = recall_score(y_test, previsoes_2, average= None) # teste de sensibilidade do modelo.


# modelo de decisão extra tree.

extra_tree_df = ExtraTreesClassifier(criterion = 'gini', max_features = None)
extra_tree_df.fit(x_train, y_train)

previsoes_3  = extra_tree_df.predict(x_test) # jogando os dados de teste para tentar prever.
previsoes_3 = extra_tree_df.predict(x_test) # teste de acurácia para ver se o modelo foi eficiente.
acuracia_3 = accuracy_score(y_test, previsoes_3) # matriz de confusão para ver a quantidade de cada informação que ficou correta.
Matriz_Confusao_3 = confusion_matrix (y_test,previsoes_3) # teste da precisão do modelo.
precision_3 = precision_score(y_test, previsoes_3, average = None) # teste da precisão do modelo.
recall_3 = recall_score(y_test, previsoes_3, average= None) # teste de sensibilidade do modelo.



# função para prever dados futuros.

def novos_dados(a):
    resultado = extra_tree_df.predict(a)
    return resultado

novos_dados = novos_dados(x_test) #testando a validade da função.


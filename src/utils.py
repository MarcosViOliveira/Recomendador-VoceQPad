import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import scale


import networkx as nx
from collections import Counter

datasets = {
  'clientes': 'Reporting - Dados Perfil de Cli',
  'ordens': 'Reporting - Dados Ordens - Reco',
  'produtos': 'Reporting - Dados Produtos - Re'
}

# Get data
def get_data(file_path1:str) -> pd.DataFrame:
  df = pd.read_excel(file_path1)
  return df

def preparation_data(df: pd.DataFrame) -> pd.DataFrame:
   # TRATAMENTO DO DATAFRAME BASE

  base = df.copy()

  # mudando de objeto para data
  base['data_compra'] = pd.to_datetime(base['data_compra'])
  base['usuário_data-cadastro'] = pd.to_datetime(base['usuário_data-cadastro'])

  # tipo de dados mudado de objeto para categoria de forma que ocupe menos espaço na memória
  base['modalidade_compra'] = base['modalidade_compra'].astype('category')
  base['plataforma-compra'] = base['plataforma-compra'].astype('category')

  # retirando os pedidos com 100% de desconto, como mencionado an dúvida 3 do projeto
  base = base.loc[(base['valor-pedido'] + base['valor-desconto'] != 0)]
  base = base.loc[(base['valor-pedido'] + base['valor-desconto']) > 0]

  # alteração de tipos de colunas
  base['qtde_itens'] = base['qtde_itens'].astype('int')
  base['dist-pedido'] = pd.to_numeric(base['dist-pedido'], errors='coerce')
  base['dist-pedido'] = base['dist-pedido'].fillna(0)

  # limitação de casas decimais
  base['dist-pedido'] = base['dist-pedido'].round(3)

  # Tirando possíveis testes, como mencionado na dúvida 4 do projeto
  base = base.loc[base['dist-pedido'] < 14.001]

  base['dias-desde-compra-anterior'].replace('-', '0', inplace=True)
  base['dias-desde-compra-anterior'] = base['dias-desde-compra-anterior'].astype(int)

  base['itens_compra'] = base['itens_compra'].apply(lambda x: x.split(' | '))
  base = base.explode('itens_compra')
  base = base.reset_index(drop=True)
  base['quantidade'] = base['itens_compra'].apply(lambda x: x[-1])
  base['itens_compra'] = base['itens_compra'].str.replace(r'\s*x\d+', '')

  base = base.rename({'id_usuário': 'user_id', 'itens_compra': 'item_id', 'data_compra': 'Date'}, axis = 1)

  return base

def get_ratings(df: pd.DataFrame) -> pd.DataFrame:
    NumProduto = pd.DataFrame(df.groupby(['user_id'])['item_id'].value_counts())
    NumProduto = NumProduto.rename(columns={'item_id': 'FreqItem'})
    NumProduto = NumProduto.reset_index(level=['user_id', 'item_id'])
    totalFreq = NumProduto.groupby(['user_id'])[['FreqItem']].sum()
    totalFreq = totalFreq.reset_index().rename({'FreqItem': 'TotalPeriodo'}, axis = 1)

    NumProduto = NumProduto.merge(totalFreq, on = 'user_id', how = 'left')
    NumProduto['FrequenciaItem'] = NumProduto['FreqItem']/NumProduto['TotalPeriodo']
    NumProduto = NumProduto.query('TotalPeriodo >= 3')
    FreqCompra = NumProduto[['user_id', 'item_id', 'FrequenciaItem']]

    dfProduto = df.copy()
    dfProduto = dfProduto.merge(FreqCompra, on = ['user_id', 'item_id'], how = 'left')
    dfProduto = dfProduto.dropna()

    df_covisitation = dfProduto[['Date', 'user_id', 'item_id', 'FrequenciaItem']].rename({'FrequenciaItem': 'rating'}, axis = 1)
    df_covisitation = df_covisitation.query('rating >= 0.1').reset_index(drop=True)

    user_list = df_covisitation['user_id'].unique()

    return dfProduto, df_covisitation, user_list

def recommend_top_n_consumptions(ratings:pd.DataFrame, n:int) -> pd.DataFrame:

    recommendations = (
        ratings
        .groupby('item_id')
        .count()['user_id']
        .reset_index()
        .rename({'user_id': 'score'}, axis=1)
        .sort_values(by='score', ascending=False)
    )

    return recommendations.head(n)

def recommend_neighbor_items(G:nx.Graph, target_id, max_recommendations=10):
      # Validando tipo do nó
      node_type = nx.get_node_attributes(G, 'node_type')[target_id]
      if node_type != 'item':
          raise ValueError('Node is not of item type.')

      # Analisando consumo dos usuários vizinhos
      neighbor_consumed_items = []
      for user_id in G.neighbors(target_id):
          user_consumed_items = G.neighbors(user_id)
          neighbor_consumed_items +=list(user_consumed_items)

      # Contabilizando itens consumidos pelos vizinhos
      consumed_items_count = Counter(neighbor_consumed_items)

      # Criando dataframe
      df_neighbors = pd.DataFrame(zip(consumed_items_count.keys(), consumed_items_count.values()))
      df_neighbors.columns = ['item_id', 'score']
      df_neighbors = df_neighbors.sort_values(by='score', ascending=False).set_index('item_id')

      return df_neighbors.head(max_recommendations)

# Classe genérica para recomendação
class CoVisitationRecommender:
  
  def __init__(self, data_total, data, item_id, user_id, rating):
    self.data_total = data_total.copy()
    self.data = data.copy()
    self.item_id = item_id
    self.user_id = user_id
    self.rating = rating
 
  def fit(self, n_most_popular=10):
    G = nx.Graph()
    G.add_nodes_from(self.data['item_id'].unique(), node_type='item')
    G.add_nodes_from(self.data['user_id'].unique(), node_type='user')
    G.add_weighted_edges_from(self.data[['item_id', 'user_id', 'rating']].values)

    items_list_ = self.data['item_id'].unique()
    users_list_ = self.data['user_id'].unique()
    users_list_ = pd.DataFrame(users_list_, columns= ['user_id']).merge(self.data_total, on = 'user_id', how = 'left')[['user_id', 'id_aplicativo']]
    app_list_ = self.data_total['id_aplicativo'].unique()

    return G, items_list_, users_list_, app_list_
  
  def get_target(self, account_id):
    result = self.data.query('user_id == @account_id')
    if result.size == 0: return None
    return result.sort_values('Date', ascending=False)['item_id'].values[0]
    
  def recommend(self, G, target_item, max_recommendations=10):
    try:
      covisitation = recommend_neighbor_items(G, target_item, max_recommendations).index
      #top = recommend_top_n_consumptions(ratings = self.rating, n = max_recommendations)
      return covisitation#, top
    except KeyError as e:
      print(f'\033[1m{target_item}\033[0;0m is not included in the recommendation matrix. Returning top 10 items:\n')
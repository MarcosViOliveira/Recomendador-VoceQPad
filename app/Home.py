import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../src')

import pickle
import pandas as pd
import streamlit as st
from utils import *

## -- Page Settings -- ##

title = "DEX VoceQPad Playground"

st.set_page_config(page_title=title)

with open ('models/recommender.pkl', 'rb') as file:
  
  # load data
  
  recommender = pickle.load(file)
  
  model, item_list, users_list, app_list = recommender.fit()

  app_number_list = pd.Series(app_list.tolist())
  
  ## -- Page Content -- ##
  
  st.title(title)
  
  ## -- Parameters -- ##
  
  users_limit = st.number_input('Número de clientes', min_value=5, max_value=100, value=5)

  app_number = st.selectbox('Selecione Aplicativo', app_number_list.tolist())

  users = pd.Series(users_list.query('id_aplicativo == @app_number')['user_id'].tolist()).drop_duplicates()
  
  account_id = st.selectbox('Selecione um cliente [Account ID]', users.head(users_limit).tolist())

  target = recommender.get_target(account_id)
  
  result = recommender.recommend(model, target)
  
  ## -- Model Result -- ##
  
  if result is not None:
    
    st.write(f'Quem comprou **{target}**, também comprou:')
    
    st.write(result.tolist())
    
  else: 
    
    st.write(f'Não há recomendações para **{target}**. Veja os mais comprados:')
    
    st.write(recommender.recommend_top_n_consumptions(recommender.get_data(), n=10).Symbol.tolist())

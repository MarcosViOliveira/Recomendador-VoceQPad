import pickle
from utils import get_data, preparation_data, get_ratings, recommend_neighbor_items, CoVisitationRecommender

# Get data
df = get_data('data/Base VoceQPad.xlsx')
# Preparation data
df = preparation_data(df =  df)
# Get implicit ratings
dfProduto, df_covisitation, user_list = get_ratings(df)

# Instiantiate recommender
recommender = CoVisitationRecommender(
    data_total = df,
    data=df_covisitation,
    item_id='item_id',
    user_id='user_id',
    rating='rating'
)

# Train recommender
model, item_list, user_list, app_list = recommender.fit()

# Save recommender
with open('models/recommender.pkl', 'wb') as model_file:
    pickle.dump(recommender, model_file)
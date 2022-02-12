import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from dateutil import parser
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
from catboost import CatBoostClassifier
from shap import TreeExplainer, summary_plot
import joblib

data = pd.read_csv("https://raw.githubusercontent.com/daggeraile/market_clustering/master/raw_data/data.csv")
original_columns = data.columns

# imputing missing data
empty_columns = data.columns[data.isnull().any()]

for column in empty_columns:
    
    if data[column].dtype == 'O':
        imputer = SimpleImputer(strategy='most_frequent')
        data[[column]] = imputer.fit_transform(data[[column]])

    if data[column].dtype == 'int' or data[column].dtype == 'float':
        imputer = SimpleImputer(strategy='mean')
        data[[column]] = imputer.fit_transform(data[[column]])
        
data = pd.DataFrame(data, columns=original_columns)

# Converting datetime features
cat_features = data.select_dtypes(include=["object"]).columns.values

def convert_datetime(feature):
    error_count = 0
    
    for i in range(5):
        try:
            parser.parse(data[feature].iloc[i])
        except ValueError:
            error_count += 1

    if error_count == 0:
        data[feature] = pd.to_datetime(data[feature])
        data[feature] = data[feature].apply(lambda x: (datetime.now() - x).days)
    
    return None

for feature in cat_features:
    convert_datetime(feature)


# onehotencoding categorical features
cat_features = data.select_dtypes(include=["object"]).columns.values
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
ohe_df = ohe.fit_transform(data[cat_features])

# creating ohe_df with column names
temp = ohe.categories_
ohe_cols = []
for i in temp:
    for j in i:
        ohe_cols.append(j)

ohe_df = pd.DataFrame(ohe_df, columns=ohe_cols)

# dropping sparse columns resulted from ohe - threshold <5% of datapoints
ohe_percent = pd.DataFrame(ohe_df.sum()/ohe_df.count(), columns=['percentage'])
sparse_columns = ohe_percent[ohe_percent['percentage']<0.05].index.values
ohe_df.drop(columns=sparse_columns, inplace=True)
ohe_cols = ohe_df.columns

# removing original columns
data.drop(columns=cat_features, inplace=True)

# concatenating both dataframe
data = pd.concat([data, ohe_df], axis=1)

# assigning first column as index
data.set_index(data.columns[0], inplace=True)

# storing original column with names
final_columns = data.columns

# scaling data with robust scaler
scaler = RobustScaler()
scaled_data = scaler.fit_transform(data)
scaled_data = pd.DataFrame(scaled_data, columns=final_columns)

if __name__=="__main__":
    joblib.dump(scaled_data, 'scaled_data.joblib')


    # performing TSNE to obtain 3 axes
    tsne = TSNE(n_components=3, verbose=1, init='pca', learning_rate='auto', random_state=42)
    axis = tsne.fit_transform(scaled_data)
    axis = pd.DataFrame(axis, columns=['x', 'y', 'z'])

    #####################
    #Start of iterations#
    #####################

    # iterating from 3 to 8 clusters
    for iteration in range(3, 8):

        # K-means clustering
        n_clusters = iteration
        model = KMeans(n_clusters=n_clusters)
        axis['cluster'] = model.fit_predict(axis)
        axis['cluster'] = axis['cluster'].apply(lambda x: x+1)
        joblib.dump(axis, f'axis_{iteration}.joblib')

        # Create cluster_average dataframe
        axis['ID'] = data.index
        axis.set_index('ID', inplace=True)
        cluster_average = data.join(axis['cluster']).groupby('cluster').agg('mean')
        size_df = pd.DataFrame(data.join(axis['cluster']).value_counts('cluster').sort_index(), columns=['size'])
        cluster_average['Size'] = size_df
        cluster_average = cluster_average.T
        cluster_average = cluster_average.round(2)
        joblib.dump(cluster_average, f'cluster_average_{iteration}.joblib')



        # converting silhouette score to layman score
        silhouette = silhouette_score(axis[['x','y','z']], axis['cluster'])
        cluster_score = round(silhouette*100)
        cluster_score
        
        # training catboostclassifier for model explanation
        model = CatBoostClassifier()
        model.fit(scaled_data, axis['cluster'])
        joblib.dump(model, f'cat_model_{iteration}.joblib')

        # explaining total contribution to cluster formation
        explainer = TreeExplainer(model)
        shap_values = explainer.shap_values(scaled_data)
        summary_plot(shap_values, scaled_data, plot_type='bar', class_names=model.classes_)

        # instantiating a dictionary to hold shap values as dataframe and names of top 5 features 
        shap_dict = {}
        for i in range(n_clusters):
            
            # storing shap values dataframe for each cluster
            shap_dict[f'cluster_{i+1}'] = pd.DataFrame(shap_values[i], columns=final_columns)

            # storing top 5 contributing features for each cluster
            shap_dict[f'top_{i+1}'] = shap_dict[f'cluster_{i+1}'].abs().sum().sort_values(ascending=False)[:5].index.values
            joblib.dump(shap_dict, f'shap_dict_{iteration}.joblib')
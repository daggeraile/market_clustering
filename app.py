import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
# import matplotlib.pyplot as plt
import numpy as np
import joblib
import plotly.graph_objects as go
from matplotlib import cm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
# from market_clustering.main import *
from sklearn.metrics import silhouette_score
from shap import TreeExplainer, summary_plot
import io
import copy
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('Market Clustering')
st.markdown("<p style='color:Green; font-size:18px;'>By Lance Ng</p>", unsafe_allow_html=True)


# uploading of data
st.markdown('## Upload Data for Clustering')
st.text_input('Enter URL for Data Upload', value='https://raw.githubusercontent.com/daggeraile/market_clustering/master/raw_data/data.csv', disabled=True)
st.write("""In the full version of this app, users will be able to upload any datasets of their own,
as long as they are reasonably cleaned. This app is able to handle numerical, categorical and even datetime features.
In addition it is also able to handle missing data through imputing. \n\n
For demonstration purpose, a sample retail dataset with 2240 unique customers has been loaded instead.\
Displaying first 5 rows of dataset below.""")

source = ""

raw_data = pd.read_csv("https://raw.githubusercontent.com/daggeraile/market_clustering/master/raw_data/data.csv")
st.dataframe(raw_data.head())

# option = st.selectbox('As this is a demonstration, please utilise either of below data instead.', ('retail customer data', 'bank customer data'))

# if option == 'retail customer data':
#     source = ""
# else:
#     source = "II"

# selecting clusters
st.markdown('## Select Number of Clusters')
st.write("""By choosing the number of clusters beforehand, we let the app handle 
the segmentation of the dataset into the desired number of clusters. We can then compare the performance 
of the segmentation model through more details below.""")

cluster_choice = st.slider(
    min_value=3, max_value=7, value=5, label=''
)

# loading axis from joblib
axis = joblib.load(f'axis_{cluster_choice}{source}.joblib')


# Converting silhouette score to layman score
silhouette = silhouette_score(axis[['x','y','z']], axis['cluster'])
cluster_score = round(silhouette*100)

st.markdown(f'### Market Segmentation Score: {cluster_score}')
st.progress(cluster_score)
st.write("""This metric measures the quality of clusters formed. Higher scores represents clearer segregations between the clusters.
Explore various number of clusters with the bar above and see how it affects the score""")

# showing plotly output
st.markdown('## Displaying Clusters')
st.write("""As we live in a 3-dimensional world, the various columns in the dataset has been compressed into three axes for visualization purpose.
Each axis consists of information captured from the various columns, and as a whole represents how the datapoints will have been clustered in a multi-dimensional space.""")
fig = px.scatter_3d(axis, x='x', y='y', z='z', color='cluster', opacity=0.7, width=800, height=800)
st.plotly_chart(fig, use_container_width=True)


st.write('\n')
st.write('\n')
st.write('\n')


# loading data from joblib
model = joblib.load(f'cat_model_{cluster_choice}{source}.joblib')
shap_dict = joblib.load(f'shap_dict_{cluster_choice}{source}.joblib')
scaled_data = joblib.load(f'scaled_data{source}.joblib')
final_columns = joblib.load(f'final_columns{source}.joblib')
cluster_average = joblib.load(f'cluster_average_{cluster_choice}{source}.joblib')
ohe_cols = joblib.load(f'ohe_cols{source}.joblib')


# explaining total contribution to cluster formation
st.markdown('## Importance of Data Point')
st.write('''The model uses the uploaded data points to meaningfully segment the market, 
however not all data points contributes equally to the formation of the clusters. The higher it appears
on this table, the more it contributed to the segmentation of the market''')
explainer = TreeExplainer(model)
shap_values = explainer.shap_values(scaled_data)
sum_plot = summary_plot(shap_values, scaled_data, plot_type='bar', class_names=model.classes_)
st.pyplot(sum_plot)

st.write('\n')
st.write('\n')
st.write('\n')

st.markdown('## Cluster Visualization')
st.write('''To help us understand each cluster more intuitively, the cluster averages of the 
important data points can be compared easily. In addition, the radar chart helps to visualize the 
difference between the cluster in relation with each other.''')

# user input to select features
temp = 0
for i in range(cluster_choice):
    temp += shap_dict[f'cluster_{i+1}'].abs().sum()

temp = temp.sort_values(ascending=False)
default_features = list(temp.index[:5])
selected_features = st.multiselect(label='Please select the features you want to include in the summary', options=final_columns, default=default_features)

st.write('\n')
st.write('\n')
st.write('\n')

# displaying cluster_average table with size and selected features
st.write("""Based on the features you have selected above, we have generated a summary table of the average values in each cluster for easy comparison.
green and red highlights represents highest and lowest averages in each selected feature.""")
display_features = ['Size'] + selected_features
st.dataframe(cluster_average.loc[display_features].style.highlight_max(axis=1, color='green').highlight_min(axis=1, color='red').format("{:.2f}"))

st.write('\n')
st.write('\n')
st.write('\n')

# Displaying overall radar chart based on selected features
st.write("""You can also compare the clusters through the radar chart below for easy visualization.
Each spoke of the chart has been scaled to a score of 1-10, to better represent the relative difference between clusters.
You may click on the cluster legends to toggle it's visibility, give it a try.
""")
scaled_average = cluster_average.T.drop(columns='Size')
random_scaler = MinMaxScaler(feature_range=(1,10))
scaled_average = random_scaler.fit_transform(scaled_average)
scaled_average = pd.DataFrame(scaled_average, columns=final_columns, index=range(1,cluster_choice+1))

radar_dict = {}
for i in range(cluster_choice):
    subset = scaled_average.loc[[i+1]]
    r = []
    for item in subset[selected_features].T.values:
        r.append(item[0])
    radar_dict[f'r_{i+1}'] = r

fig = go.Figure()
for i in range(1, cluster_choice+1):

    fig.add_trace(go.Scatterpolar(
          r=radar_dict[f'r_{i}'],
          theta=selected_features,
          fill='toself',
          name=f'cluster_{i}'
    ))

fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0, 10]
    )),
  showlegend=True
)

st.plotly_chart(fig)

st.write('\n')
st.write('\n')
st.write('\n')


st.markdown('## Cluster Summary')
st.write('''To help us understand the key characteristic of each cluster, the beeswarm
charts indicates the respective SHAP values of each data point. Simply put, the data points
are arranged by their importance with regards to the formation of the cluster. Automated layman
sentences are generated for further explanation.''')

# explaining cluster through beeswarm and layman description
for i in range(cluster_choice):
    st.markdown(f"### Summary for Cluster {i+1}")
    st.pyplot(summary_plot(shap_values[i], scaled_data, max_display=10))
    for feature in shap_dict[f'top_{i+1}']:
        if feature in ohe_cols:
            if scaled_data[feature].corr(shap_dict[f'cluster_{i+1}'][feature])>0:
                st.write(f'Members of this cluster likely to have or be "{feature}"')
            else: 
                st.write(f'Members of this cluster likely not to have or be "{feature}"')
        else:
            if scaled_data[feature].corr(shap_dict[f'cluster_{i+1}'][feature])>0:
                st.write(f'Members of this cluster likely to have higher "{feature}"')
            else: 
                st.write(f'Members of this cluster likely to have lower "{feature}"')
    st.write('---')
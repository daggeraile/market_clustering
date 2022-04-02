# Market Clustering - Tool for Marketers
An automated clustering tool that converts customer database from spreadsheets to meaningful customer clusters.

To see the app in deployment, please visit https://market-clustering.herokuapp.com/
Kindly note that the app may take several minutes to load in full.
## About Market Clustering
This tool was built for marketers to uncover hidden customer segments within their customer database.
With the use of boosted tree model and SHAP explainer tool, we demystify unsupervised machine learning model.
The resulting customer segments can be broken down into it's key differentiating attributes.

## Data Input
The model is built to take in a wide range of customer attributes as it's input to a reasonable extend.
It is able to handle missing data, auto-detect datetime features and of course the usual categorical and numerical data.
After initial data cleaning, the dataset is scaled with RobustScaler so as it drastically improves performance of clustering models.

## Unsupervised Learning
We then reduce the dimension of the dataset by applying the t-distributed stochastic neighbor embedding (T-SNE) principle.
We employ T-SNE instead of PCA as customer data can possess non-linear relationships.
The next step is to fit the resultant dataset into K-Means clustering model which groups data points by euclidean distance.

## Explaining Output
To ensure the output is explanable, we fit a gradient boosted tree model with the initial dataset & cluster prediction.
This ensures the tree model learn the importance of each attribute captured by it's various node splits.
Lastly, by use of SHAP explainer tool, we are able to calculate and visualize the key attributes contributing to each cluster's formation.
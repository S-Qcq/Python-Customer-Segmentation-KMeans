# Python-Customer-Segmentation-KMeans

This project segments customers of a retail company into groups with similar behavior, using K-Means clustering, and wraps the trained model in a small Streamlit app that predicts the segment of a new customer.

Dataset: Customer Segmentation - Clustering (Kaggle)

## Project overview

The notebook customer_segmentation.ipynb covers the following steps:

- Data cleaning and exploration. Quick cleaning (date conversion, imputation of missing income values with the median, handling of invalid marital status categories) and creation of new variables such as age, age group and total spending. Exploratory analysis of the distributions and of the relationships between income, spending, education, age and marital status.
- K-Means clustering. Selection of seven customer features, standardization with StandardScaler, and choice of the number of clusters with the elbow method.
- Cluster interpretation. Each cluster is profiled using its average values (age, income, spending, store and web purchases, web visits, recency) and described in business terms.
- PCA for visualization. A Principal Component Analysis reduces the features to two components, used only to plot the clusters on a 2D scatterplot. The loadings are inspected to interpret the two axes. PCA is not used for the clustering itself.
- Model export. The trained K-Means model and the scaler are saved with joblib (kmeans_model.pkl and scaler_model.pkl).

The Streamlit app (app.py) loads the saved model and scaler. The user enters the data of a hypothetical new customer and the app returns the predicted cluster, together with its description and a suggested action.


## Requirements
- Python 3.14 was used to develop and run the project. Other versions may work, but the pinned library versions below were installed with Python 3.14.
- Libraries. All the required libraries and their exact versions are listed in requirements.txt. 
- tkinter. The notebook uses tkinter to open a file chooser for the CSV file. tkinter is not a pip package, so it is not listed in requirements.txt: it is part of the Python standard library and is included in the standard Python installer for Windows and macOS. On Linux, install it with sudo apt install python3-tk.

## Notes
The clustering uses random_state=42, so results are reproducible with the same library versions. The cluster numbers in the descriptions of the notebook and in app.py refer to the saved model: if you retrain the model, check the cluster profiles again and update the descriptions.
Clusters are described from average values only, and the interpretations are only hypothetical.


# download iris from sklearn and print it
import numpy as np
from sklearn import datasets
import pandas as pd

iris = datasets.load_iris()
# print(iris)

# get first column
print(iris.target)

# divide the firs column into 4 parts
percentiles = np.percentile(iris.data[:, 0], [25, 50, 75, 100])
print(percentiles)

# load iris into pandas
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
df_iris['target'] = iris.target
print(df_iris)

# sort by sepal length
df_iris.sort_values(by='sepal length (cm)', inplace=True)
print(df_iris)

# create 4 dataframes based on sepal length vales (percentiles)
df_iris_1 = df_iris[df_iris['sepal length (cm)'] <= percentiles[0]]
df_iris_2 = df_iris[(df_iris['sepal length (cm)'] > percentiles[0]) & (df_iris['sepal length (cm)'] <= percentiles[1])]
df_iris_3 = df_iris[(df_iris['sepal length (cm)'] > percentiles[1]) & (df_iris['sepal length (cm)'] <= percentiles[2])]
df_iris_4 = df_iris[df_iris['sepal length (cm)'] > percentiles[2]]

# calculate counts of each species in each dataframe
df_iris_counts_1 = df_iris_1['target'].value_counts()
df_iris_counts_2 = df_iris_2['target'].value_counts()
df_iris_counts_3 = df_iris_3['target'].value_counts()
df_iris_counts_4 = df_iris_4['target'].value_counts()

# create a dataframe with counts of each species in each dataframe
df_iris_counts = pd.DataFrame([df_iris_counts_1, df_iris_counts_2, df_iris_counts_3, df_iris_counts_4])
print(df_iris_counts)

# calculate conditional probabilities
df_iris_probs = df_iris_counts.div(df_iris_counts.sum(axis=1), axis=0)
print(df_iris_probs)


entropy = 0
# calculate entropy for each row (sum of -p*log2(p)
for row in df_iris_probs.iterrows():
    print(row[1])
    print(-np.sum(row[1]*np.log2(row[1])))
    entropy += -np.sum(row[1]*np.log2(row[1])) * np.sum(row[1])

print(entropy/4)




import pandas as pd
import numpy as np
from IPython.display import display_html
from sklearn import cluster

def toy_dataset():
    ratings = [['Lokesh',5,5,2,1],['Jyoti',4,5,3,2],['Bijay',4,4,4,3],['Sita',2,2,4,5],['Manish',1,2,3,4],['Ram',2,1,5,5]]
    titles = ['user','loot','chino','Ghar','Aatma']
    movies = pd.DataFrame(ratings,columns=titles)
    display_html(movies)
    return movies

def k_means_learn(k,movies):
    data = movies.drop('user',axis=1)
    k_means = cluster.KMeans(n_clusters=2,max_iter=50,random_state=1)
    k_means.fit(data)
    labels = k_means.labels_
    pd.DataFrame(labels,index=movies.user,columns=['ClusterID'])
    print('Learned cluster centroids for two clusters 0 and 1:')
    centroids = k_means.cluster_centers_
    display_html(pd.DataFrame(centroids,columns=data.columns))
    print('Now you can use cluster centroids to other users to determine their cluster assignments.')
    return(k_means)

def cluster_new_data(k_means,movies):
    testData = np.array([[4,5,1,2],[3,2,4,4],[2,3,4,1],[3,2,3,3],[5,4,1,4]])
    labels = k_means.predict(testData)
    labels = labels.reshape(-1,1)
    username = np.array(['Radhe','Riya','Pratik','Aayaan','Shyam']).reshape(-1,1)
    cols = movies.columns.tolist()
    newusers = pd.DataFrame(np.concatenate((username,testData),axis=1),columns=cols)
    cols.append('Assigned Cluster')
    newusers_cluster = pd.DataFrame(np.concatenate((username,testData,labels),axis=1),columns=cols)
    print('Your new users(test data) are:')
    display_html(newusers)
    print('New Users with their assigned cluster:')
    display_html(newusers_cluster)

def main():
    k = 2
    movies = toy_dataset()
    k_means = k_means_learn(k,movies)
    cluster_new_data(k_means,movies)

main()
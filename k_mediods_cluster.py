import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.datasets import make_blobs
from IPython.display import display_html


def toy_dataset():
    centers = [[1,1],[-1,-1],[1,-1]]
    X,labels_true = make_blobs(n_samples=750,centers=centers,cluster_std=0.4,random_state=42)
    titles = ['x1','x2']
    data = pd.DataFrame(X,columns=titles)
    print("Do you want to view first 10 data elements")
    choice = input()
    if choice=='yes':
        display_html(data.head(10))
    return data,X
def K_medoids(data):
    k_medoids=KMedoids(n_clusters=3,max_iter=50).fit(data)
    labels=k_medoids.labels_
    print("learned cluster centroids for three cluseters")
    centroids = k_medoids.cluster_centers_
    display_html(pd.DataFrame(centroids,columns=data.columns))
    return k_medoids,labels

def cluster_new_data(k_medoids):
    testData=np.array([[0.81,1.12],[-1.145,-1.194],[0.676,0.7133],
    [0.442,-1.3245],[1.23623,1.34635],[-0.93423,0.0332],[-1.00234,-1.546],
    [0.946,-0.467],[1.534,0.4789],[1.23523,1.0547]])
    labels=KMedoids.predict(k_medoids,testData)
    labels=labels.reshape(-1,1)
    cols=['x1','x2']
    cols.append("Assigned Cluster")
    newdata_cluster=pd.DataFrame(np.concatenate((testData,labels),axis=1),columns=cols)
    display_html(newdata_cluster)

def view_cluster(labels,X,k_medoids):
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0,1,len(unique_labels))]
    for k,col in zip(unique_labels,colors):
        class_member_mask = labels==k
        xy=X[class_member_mask]
        plt.plot(xy[:,0],xy[:,1],'o',markerfacecolor=tuple(col),markeredgecolor='k',markersize=6)
        plt.plot(k_medoids.cluster_centers_[:,0],k_medoids.cluster_centers_[:,1],'o',markerfacecolor="cyan",markeredgecolor='k',markersize=6)
        plt.title("KMedoids Clustering. Medoids are represented in cyan")

def main():
    data,X=toy_dataset()
    k_medoids,labels=K_medoids(data)
    print("Do you want to view the scatter plot of learned cluster? ")
    choice=input()
    if choice=='yes':
        view_cluster(labels,X,k_medoids)
    print("Do you want clustering for new data based on learned cluster?")
    choice=input()
    if choice=='yes':
        cluster_new_data(k_medoids)
    else:
        quit()
        
main()

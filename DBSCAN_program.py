from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
import IPython
from IPtyhon.display import dislay_html

def toy_dataset():
    value = np.array([[1,3],[4,2],[2,3],[6,7],[8,9],[10,12],[10,15],
    [50,19],[20,20],[44,21],[46,18],[51,19],
    [50,10],[40,22],[50,23],[49,53],[50,25],
    [50,58],[40,49],[50,55],[49,53],[50,52],
    [25,80],[100,30],[150,90]])
    titles = ['x','y']
    data=pd.DataFrame(value,columns=titles)
    print("Do you want to view data")
    option=input()
    if option=='yes':
        print("First five data points")
        dislay_html(data.head())
    print("Do you want to view scatter plot of data")
    option=input()
    if option=='yes':
        print("Data points scatter plot")
        data.plot.scatter(x='x',y='y')
    return data

def Dbscan_clustering(data):
    db=DBSCAN(eps=10.5,min_samples=4).fit(data)
    core_samples_mask = np.zeros_like(db.labesl_,dtype=bool)
    core_samples_mask[db.core_sample_indices_]=True
    labels = pd.DataFrame(db.labels_,columns=['ClusterID'])
    result = pd.concat((data,labels),axis=1)
    result.plot.scatter(x='x',y='y',c='Cluster ID',colormap='jet')

def main():
    data=toy_dataset()
    print("Cluster contructed by DBSCAN:")
    Dbscan_clustering(data)
main()
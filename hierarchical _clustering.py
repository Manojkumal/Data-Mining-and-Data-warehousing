import pandas as pd
from IPython.display import display_html
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

def toy_dataset():
    animal=[['human',1,1,0,0,1,0,'mammals'],['python',0,0,0,0,0,1,'reptiles'],
    ['salmon',0,0,1,0,0,0,'fishes'],['whale',1,1,1,0,0,0,'mammals'],
    ['frog',0,0,1,0,1,1,'amphibians'],['komodo',0,0,0,0,1,0,'reptiles'],
    ['bat',1,1,0,1,1,1,'mammals'],['pigeon',1,0,0,1,1,0,'birds'],
    ['cat',1,1,0,0,1,0,'mammals'],['leopard shark',0,1,1,0,0,0,'fishes'],
    ['turtle',0,0,1,0,1,0,'reptiles'],['penguin',1,0,1,0,1,0,'birds'],
    ['prcupine',1,1,0,0,1,0,'mammals'],['eel',0,0,1,0,0,0,'fishes'],
    ['salanmander',0,0,1,0,1,1,'amphibians']]
    titles=['Name','Warm_blooded','Give_birth','Aquatic_creature','Aerial_reature','Has_legs','Hibernates','Class']
    data = pd.DataFrame(animal,columns=titles)
    print("Do you wnat to view data")
    choice=input()
    if choice=='yes':
        display_html(data)
    return data

def ward(names,X,Y):
    Z=hierarchy.linkage(X.to_numpy(),'ward')
    dn=hierarchy.dendrogram(Z,labels=names.tolist(),orientation='right')

def centroid(names,X,Y):
    Z=hierarchy.linkage(X.to_numpy(),'centroid')
    dn=hierarchy.dendrogram(Z,labels=names.tolist(),orientation='right')

def group_average(names,X,Y):
    Z=hierarchy.linkage(X.to_numpy(),'average')
    dn=hierarchy.dendrogram(Z,labels=names.tolist(),orientation='right')

def complete_link(names,X,Y):
    Z=hierarchy.linkage(X.to_numpy(),'complete')
    dn=hierarchy.dendrogram(Z,labels=names.tolist(),orientation='right')

def single_link(names,X,Y):
    Z=hierarchy.linkage(X.to_numpy(),'single')
    print("Dendrogarm of single link Hierachical clustering")
    dn=hierarchy.dendrogram(Z,labels=names.tolist(),orientation='right')


def main():
    data = toy_dataset()
    names=data['Name']
    Y=data['Class']
    X = data.drop(['Name','Class'],axis=1)
    print(X.head())
    print('Your data is ready!')
    print("select your option':\
        1.single_link\
        2.Complete_link\
        3.Group_average\
        4.Centroid\
        5.ward")
    choice=int(input())
    if choice==1:
        single_link(names,X,Y)
    elif choice==2:
        complete_link(names,X,Y)
    elif choice==3:
        group_average(names,X,Y)
    elif choice==4:
        centroid(names,X,Y)
    else:
        print("Enter correct choice next time")
        quit()

main()
   


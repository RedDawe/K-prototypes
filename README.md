# K prototypes

Clustering analysis of a subset of the adult dataset.

First i plot losses over over different number of clusters to figure out how many clusters  I want.

![losses](https://i.imgur.com/3G2ZPfg.png)

Using the elbow method I have two choices. I went with 4.

![output](https://i.imgur.com/S932rja.png)

These are our 4 cluster (stereotypical types of people). Only the last one earns over 50k. (which was the original point of the dataset)

![TSNE](https://i.imgur.com/DuwswAb.png)

Finally, here is a TSNE visualization

Z. Huang. Extensions to the k-means algorithm for clustering large data sets with categorical values. Data Mining and Knowledge Discovery, 2(3):283–304, 1998

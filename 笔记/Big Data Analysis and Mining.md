![image.png](https://upload-images.jianshu.io/upload_images/665202-bb94c950d325d43f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

<!--more-->
csdn：https://blog.csdn.net/linxid/article/details/80494922

Github Page：https://linxid.github.io/

# Chapter1 Introduction
## 1.1 What is Big Data:

**Answer：** used to describe a massive structured and unstructured data that is so large that it is difficult to process using traditional database and software techniques.

## 1.2 4V of Bid Data:
**Answer：** Volume:大量; Velocity:快速；Variety:多样; Veracity:真实准确。

## 1.3 What is Data mining:
**Answer：** under acceptable computational efficiency limitations, applying data analysis
and discovery algorithms, to produce a particular enumeration of patterns over the
data

## 1.4 Main Data Mining Tasks:
**Answer：** Association rule mining,cluster analysis,classification,prediction,outlier detection

# Chapter2 Basic Concepts

## 2.1 Tasks of ML:
**Answer：** supervised learning; Unsupervised learning; Semi-supervised learning
Overfitting, Underfitting

## 2.2 How to avoid Overfitting:
**Answer：** Increase Sample;Remove outliers;Decrease model complecity,train-validation-test
(cross validation),regularization

## 2.3 Basic Algorithm：
### 2.3.1 Classification:
KNN；Naive Bayes；Decision Tree；SVM；

#### 2.3.2 Ensemble Learning：
Bagging -> Random Forest；Boosting -> AdaBoost；Stacking；

### 2.3.3 Clustering:
K-means；Hierarchical Clustering；DBSCAN；Apriori;

# Chapter3 Hashing
## Why we need Hashing?
To resolve the challenge, like the curse of dimensionality, storage cost, and query speed.

## 3.1 Find Similar Items

### 3.1.1 Shingling
* k-Shingling

### 3.1.2 Minhashing

**Definition:** the number of the first row in which column

* Jaccard Similarity of Sets
* From sets to Boolean Matrices
* Signatures --> Signature Matrix
* Hashing Function

**How to compute the Signature matrix**

### 3.1.3 Locality Sensitive Hashing(LSH)


### References：
https://blog.csdn.net/linxid/article/details/79745964

# Chapter4 Sampling

# Chapter5 Data Stream
## 5.1 What is Data Stream

### what is the challenge of the Data Stream:
* Single Pass Handling
* Memory limitation
* Low Time complexity
* Concept Drift

## 5.2 What is Concept Drift
Concept drift means that the statistical properties of the target variable, which the model is trying to predict, change over time in unforeseen ways.

**the probability distribution changes.**

**Real concept drift:**
p(y|x) changes

**Virtual concept drift:**
p(x)changes,but not p(y|x)

## 5.3 Concept drift detection:
### 1.distribution-based detector
Monitoring the change of data distributions
#### Disadvantage:
* Hard to determine window size
* learn concept drift slower
* virtual concept drift

### 2.Error-rate based detector:
based on the change of the classification performance.
#### Disadvantage:
* Sensitive to noise
* Hard to deal with gradual concept drift
* Depend on learning model itself heavily

### Drift detection method:DDM

## 5.4 Data Stream Classification:
#### Data stream classification circle:
* Process an example at a time, and inspect it only once
* Be ready to predict at any point
* Use a limited amount of memory
* Work in a limited amount of time

### 5.4.1 VFDT(Very Fast Decision Tree)
**Algorithm:**
* calculate the information gain for the attributes and determines the best two attributes
* At each node,check for the condition: delta(G) = G(a) - G(b) > e
* if condition satisfied, create child nodes based on the test at the node.
* if not, stream in more examples and perform calculations till condition satisfied.

**Strengths:**
* Scale better than traditional methods
* incremental

**Weakness:**
* Could spend a lot of time with times
* Memory used with tree expansion
* Number of candidate a

## 5.5 Data Stream Clustering:
### Data stream clustering Framework:
#### Online Phase:
Summarize the data into memory-efficient data structures
#### Offline Phase:
Use a clustering algorithm to find the data partition

### References：
https://machinelearningmastery.com/gentle-introduction-concept-drift-machine-learning/
http://www.liaad.up.pt/kdus/products/datasets-for-concept-drift
https://www.hindawi.com/journals/tswj/2015/235810/

# Chapter6 Graph Mining
## 6.1 BackGround:
### Applications in real-world:
* information Maximization
* computer network security
* prediction

### Network types:
* regular
* Random
* small world
* scale free

### Six degree of separation:
The average distance between two random individuals in the USA: 6

The average distance between two randomly users in Facebook(721 million active users, 69 billion links): 4.74

## 6.2 Key Node Identification

### 6.2.1 Centrality:
* Degree Centrality
* Between Centrality
* Closeness Centrality

### 6.2.2 K-shell Decomposition:
Advantage:
* Low computational complexity
* Reveal the hierarchy structure clearly

Disadvantage:
* Can not use in a lot of networks
* Too coarse, sometimes is inferior to degree measure.

[Explation:](https://www.youtube.com/watch?v=6Mk9NnboDsQ)

Prune all the nodes with degree 1 till no degree 1 nodes left in the network, the nodes pruned have ks=1. Similarly, prune other nodes having degree 2 and assign them ks =2. Repeat, till the graph becomes empty.

### 6.2.3 PageRank:

If a page is linked with many high-cited pages, then it will gain high PageRank score.
We assume a customer can use URL to link to any pages, to solve the problem that a node has no outlinks.

the equation of a Page's PR:
![image.png](https://upload-images.jianshu.io/upload_images/665202-b964a4bd418b6359.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**Explanation:**

http://blog.jobbole.com/23286/

https://www.cnblogs.com/rubinorth/p/5799848.html

https://en.wikipedia.org/wiki/PageRank#Algorithm
## 6.3 Community Detection
#### How to find intrinsic Community structure in large-scale networks:
* **Minimum cut:**
may return an imbalanced partition.

* **Ratio Cut & Normalized cut:**
  How to calculate Ratio Cut and Normalized Cut. We can use a spectral clustering algorithm to calculate it.
* **Modularity Maximization:**
measure the strength of a community by taking into account the degree distribution.

#### A new viewpoint for community detection
### References：
http://blog.sciencenet.cn/blog-3075-982948.html

# Chapter7 Hadoop-Spark

#k means clustring

from sklearn.cluster import KMeans

X = [[10],[14],[18],[97],[85],[90],[45]]

model = KMeans(n_clusters = 2,random_state = 42,n_init=10)

model.fit(X)

print(model.labels_)

print(model.predict([[20]]))


#knn 

from sklearn.neighbors import KNeighborsClassifier

X = [[10],[12],[15],[80],[85],[90]]

Y = ["Low","Low","Low","High","High","High"]

model = KNeighborsClassifier(n_neighbors = 3)

model.fit(X,Y)

prediction = model.predict([[20]])

print("Prediction:",prediction[0])
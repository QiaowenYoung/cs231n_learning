Notes
=====
Details about this assignment can be found [on the course webpage](http://cs231n.github.io/), under Assignment #1 of Spring 2019.<br>
Run on anaconda. Need to modify the 7th row of [data_utils.py](https://github.com/QiaowenYoung/cs231n_learning/blob/master/assignment1/cs231n/data_utils.py) to `import imageio` as I did.
## [knn](https://github.com/QiaowenYoung/cs231n_learning/blob/master/assignment1/knn.ipynb)
### [compute distances in 2/1/0 loops](https://github.com/QiaowenYoung/cs231n_learning/blob/master/assignment1/cs231n/classifiers/k_nearest_neighbor.py)
```Python
dists[i][j] = np.sqrt(np.sum((X[i] - self.X_train[j]) ** 2))
dists[i] = np.sqrt(np.sum((X[i] - self.X_train) ** 2, axis = 1))
```
[np.sum() function](https://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html), the second argument is to assign axis along which the sum op is on<br>
```Python
dists += np.sum(self.X_train ** 2, axis = 1).reshape(1, num_train)
dists += np.sum(X ** 2, axis = 1).reshape(num_test, 1)
dists -= 2 * np.dot(X, self.X_train.T)
dists = np.sqrt(dists)
```
Refer to [broadcast](https://www.runoob.com/numpy/numpy-broadcast.html)<br>
### [predict_labels](https://github.com/QiaowenYoung/cs231n_learning/blob/master/assignment1/cs231n/classifiers/k_nearest_neighbor.py)
```Python
closest_y = self.y_train[np.argsort(dists[i])[0:k]]
y_pred[i] = np.argmax((np.bincount(closest_y)))
```
[np.argsort](https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html) returns the indices that would sort an array<br>
[np.argmax](https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html) returns the indices of the maximum values along an axis<br>
[np.bincount](https://docs.scipy.org/doc/numpy/reference/generated/numpy.bincount.html) returns an array where the ith element represents the occurences of i
### [cross validation](https://github.com/QiaowenYoung/cs231n_learning/blob/master/assignment1/knn.ipynb)
```Python
X_t = np.vstack(X_train_folds[0: i] + X_train_folds[i + 1: ])
y_t = np.hstack(y_train_folds[0: i] + y_train_folds[i + 1: ])
```
[np.vstack](https://docs.scipy.org/doc/numpy/reference/generated/numpy.vstack.html) can concatenate arrays row wisely<br>
[np.hstack](https://docs.scipy.org/doc/numpy/reference/generated/numpy.hstack.html) column wisely
Maybe can also refer to [np.concatenate](https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html), but I failed and idky

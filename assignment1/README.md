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
## [svm](https://github.com/QiaowenYoung/cs231n_learning/blob/master/assignment1/svm.ipynb)
### [svm_loss_naive](https://github.com/QiaowenYoung/cs231n_learning/blob/master/assignment1/cs231n/classifiers/linear_svm.py)
Look up [notes](https://cs231n.github.io/optimization-1/) to get `loss` and `dW`
### [svm_loss_vectorized](https://github.com/QiaowenYoung/cs231n_learning/blob/master/assignment1/cs231n/classifiers/linear_svm.py)
```Python
margin = scores - scores[range(N), y].reshape(-1, 1) + 1
margin[range(N), y] = 0 # make the margin[y[i]] 0
margin = np.maximum(margin, 0) # make the negative elements in margin 0
```
[array.reshape(-1, 1)](https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape) will cause the array's shape to be (N,), and the new array is N * 1; reshape(-1) is 1 * N
```Python
counts = (margin > 0).astype(int)
# counts is of the same shape as margin, and counts[i][j] is 1 for margin[i][j] > 0, 0 for margin[i][j] <= 0
counts[range(N), y] = - np.sum(counts, axis = 1)
dW += np.dot(X.T, counts) / N + reg * W
```
### [linear_classifier](https://github.com/QiaowenYoung/cs231n_learning/blob/master/assignment1/cs231n/classifiers/linear_classifier.py)
Look up [notes](https://cs231n.github.io/optimization-1/) to implement descent gradient
## [softmax](https://github.com/QiaowenYoung/cs231n_learning/blob/master/assignment1/softmax.ipynb)
### [softmax_loss_naive](https://github.com/QiaowenYoung/cs231n_learning/blob/master/assignment1/cs231n/classifiers/softmax.py)
Look up [notes](https://cs231n.github.io/linear-classify/#softmax) to get improved implementation<br>
Note that new `dW` will be calculated based on new `L(W)` function
### [softmax_loss_vectorized](https://github.com/QiaowenYoung/cs231n_learning/blob/master/assignment1/cs231n/classifiers/softmax.py) & [tune hyperparameters](https://github.com/QiaowenYoung/cs231n_learning/blob/master/assignment1/softmax.ipynb)
You just basically inherit codes from svm

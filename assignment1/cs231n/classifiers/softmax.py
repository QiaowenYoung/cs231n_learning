from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    C = W.shape[1]
    N = X.shape[0]
    for i in range(N):
        scores = X[i].dot(W)
        scores -= np.max(scores)
        exp_sum = np.sum(np.exp(scores))
        loss += -scores[y[i]] + np.log(exp_sum)
        dW[:, y[i]] -= X[i]
        for j in range(C):
            dW[:, j] += (np.exp(scores[j]) / exp_sum) * X[i]
        
    loss /= N
    dW /= N
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    N = X.shape[0]
    scores = X.dot(W)
    scores -= scores.max(axis = 1).reshape(-1, 1)
    exp_sum = np.sum(np.exp(scores), axis = 1)
    margin = -scores[range(N), y] + np.log(exp_sum)
    loss += np.sum(margin)
    
    scores = np.exp(scores) / exp_sum.reshape(-1, 1)
    scores[range(N), y] -= 1
    dW += (X.T).dot(scores)
    
    loss /= N
    dW /= N
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

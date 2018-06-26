import numpy as np
from random import shuffle

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
  num_class = W.shape[1]
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
    scores = X[i].dot(W)
    scores -= np.max(scores)
    scores = np.exp(scores)/np.sum(np.exp(scores))
    loss_h = -np.log(scores[y[i]])
    loss += loss_h
    for j in range(num_class):
        if j == y[i]:
            dW[:,j] += (scores[j] - 1) * X[i]
        else:
            dW[:,j] += scores[j] * X[i]
    
  loss = loss/num_train +0.5*reg*np.sum(W*W)
  dW = dW/num_train + reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_class = W.shape[1]
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  scores -= np.max(scores).reshape((-1,1))   #shifted
  softmax_output = np.exp(scores)/(np.sum(np.exp(scores),axis =1).reshape((-1,1)))   #求导时只需真实值位置-1即可
  loss = -np.log(softmax_output[np.arange(num_train),y])
  loss = np.sum(loss)/num_train + 0.5*reg * np.sum(W * W)
  dS = softmax_output                         #BP到softmax的值
  dS[np.arange(num_train),y] -= 1
  dW = np.dot(X.T,dS)
  dW = dW/num_train + reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


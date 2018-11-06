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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  dim = X.shape[1]

  loss = 0.0
  dW = np.zeros_like(W)
  probs = np.zeros((num_train, num_classes))

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
    for j in range(num_classes):
      for k in range(dim):
        probs[i,j] += W[k,j] * X[i,k]
    probs[i, :] = np.exp(probs[i, :])
    probs[i, :] /= np.sum(probs[i, :])

  loss -= np.sum(np.log(probs[np.arange(num_train), y]))
  loss /= num_train
  loss += reg * np.sum(W*W)
  pass
  probs[np.arange(num_train), y] -= 1 

  for i in range(num_train):
    for j in range(dim):
      for k in range(num_classes):
        dW[j, k] += X[i, j] * probs[i, k] 
  
  dW /= num_train
  dW += reg * W
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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  dW = np.zeros_like(W)
  probs = np.zeros((num_train, num_classes))
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  probs = X.dot(W)
  probs = np.exp(probs)
  probs /= np.sum(probs, axis=1, keepdims=True)
  loss -= np.sum(np.log(probs[np.arange(num_train), y]))
  loss /= num_train
  loss += reg * np.sum(W*W)

  #dw
  probs[np.arange(num_train), y] -= 1 
  dW += X.T.dot(probs) 
  dW /= num_train
  dW += reg * W

  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


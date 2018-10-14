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
  for i in range(X.shape[0]):
    scores =  X[i,:].dot(W) #(1, C)
    shift_scores = scores - np.max(scores)             # simplify your calculations
    tmp = np.exp(shift_scores[y[i]]) / np.sum(np.exp(shift_scores))
    loss += -np.log(tmp)
    for j in range(W.shape[1]):
        # (1, C)
        y_pred = np.exp(shift_scores) / np.sum(np.exp(shift_scores)).reshape([1,-1])
        #print(y_pred[0,j].shape, X[i,:].T.shape, dW[:,j].shape)
        if j != y[i]:
            dW[:,j] += y_pred[0,j]*X[i,:].T
        else:
            dW[:,j] += (y_pred[0,j]-1)*X[i,:].T
  loss /= X.shape[0]
  # do not forget regulazation:
  loss += reg*0.5*np.sum(W * W)
  dW = dW + reg*W
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W) # (N, C)
  shift_scores = scores - np.max(scores, axis=1).reshape([-1,1])
  softmax_output = np.exp(shift_scores) / np.sum(np.exp(shift_scores), axis=1).reshape([-1,1])
  loss = np.sum(-np.log(softmax_output[np.arange(num_train), y]))
  loss /= num_train 
  loss +=  0.5* reg * np.sum(W * W)

  dS = softmax_output.copy()
  dS[range(num_train), list(y)] += -1
  dW = X.T.dot(dS)
  dW = dW + reg* W 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


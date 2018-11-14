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
  

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_training = X.shape[0]
  data_loss = 0.0
  dscores = np.zeros((num_training, W.shape[1]))
  
  for i in range(num_training):
    
    f = X[i].dot(W) #class scores
    #subtract each score from the maximum class scores for numerical stability
    shifted_f = f - np.max(f)    
    exp_scores = np.exp(shifted_f)
    probs = exp_scores/np.sum(exp_scores)
    dscores[i:] = probs

    data_loss += -np.log(probs[y[i]])         
        
  data_loss /= num_training 
  reg_loss = 0.5*reg*np.sum(W*W)
  loss = data_loss + reg_loss


  # compute the gradient on scores
  dscores[range(num_training),y] -= 1
  dscores /= num_training
  dW = np.dot(X.T, dscores)
  dW += reg*W
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
  num_training = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  shifted_scores = scores - np.max(scores)
  exp_scores = np.exp(shifted_scores)  
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

  # compute the loss: average cross-entropy loss and regularization
  correct_logprobs = -np.log(probs[range(num_training),y])
  data_loss = np.sum(correct_logprobs)/num_training
  reg_loss = 0.5*reg*np.sum(W*W)
  loss = data_loss + reg_loss

  # compute the gradient on scores
  dscores = probs
  dscores[range(num_training),y] -= 1
  dscores /= num_training
  
  # backpropate the gradient to the parameters (W,b)
  dW = np.dot(X.T, dscores)
    
  dW += reg*W # regularization gradient
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


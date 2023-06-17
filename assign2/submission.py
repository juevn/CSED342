#!/usr/bin/python

import random
import collections
import math
import sys
from collections import Counter
from util import *


############################################################
# Problem 1: hinge loss
############################################################

def problem_1a():
    """
    return a dictionary that contains the following words as keys:
        so, touching, quite, impressive, not, boring
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    worddict = {
        'so': 1,
        'touching': 1,
        'quite': 0,
        'impressive': 0,
        'not': -1,
        'boring': -1,
    }

    return worddict
    # END_YOUR_ANSWER

############################################################
# Problem 2: binary classification
############################################################

############################################################
# Problem 2a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_ANSWER (our solution is 6 lines of code, but don't worry if you deviate from this)
    wordlist = x.split()
    feat = dict()
    for word in wordlist:
        if word not in feat:
            feat[word] = 1
        else:
            feat[word] += 1

    return feat
    # END_YOUR_ANSWER

############################################################
# Problem 2b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note:
    1. only use the trainExamples for training!
    You can call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    2. don't shuffle trainExamples and use them in the original order to update weights.
    3. don't use any mini-batch whose size is more than 1
    '''
    weights = {}  # feature => weight

    def sigmoid(n):
        return 1 / (1 + math.exp(-n))

    # BEGIN_YOUR_ANSWER (our solution is 14 lines of code, but don't worry if you deviate from this)
    def loss_deriv(feaext,y):
        return sigmoid(-y * dotProduct(weights, feaext)) * (-y)
    
    for _ in range(numIters):
        for x, y in trainExamples:
            increment(weights, -eta*loss_deriv(featureExtractor(x),y), featureExtractor(x))
    

    # END_YOUR_ANSWER
    return weights

############################################################
# Problem 2c: ngram features

def extractNgramFeatures(x, n):
    """
    Extract n-gram features for a string x
    
    @param string x, int n: 
    @return dict: feature vector representation of x. (key: n consecutive word (string) / value: occurrence)
    
    For example:
    >>> extractNgramFeatures("I am what I am", 2)
    {'I am': 2, 'am what': 1, 'what I': 1}

    Note:
    There should be a space between words and NO spaces at the beginning and end of the key
    -> "I am" (O) " I am" (X) "I am " (X) "Iam" (X)

    Another example
    >>> extractNgramFeatures("I am what I am what I am", 3)
    {'I am what': 2, 'am what I': 2, 'what I am': 2}
    """
    # BEGIN_YOUR_ANSWER (our solution is 12 lines of code, but don't worry if you deviate from this)

    ngram_feature = dict()
    wordlist = x.split()
    for i in range(len(wordlist) - n + 1):
        word = ' '.join(wordlist[i:i+n])
        if word not in ngram_feature:
            ngram_feature[word] = 1
        else:
            ngram_feature[word] += 1
    # END_YOUR_ANSWER
    return ngram_feature

############################################################
# Problem 3a: k-means exercise
############################################################

def problem_3a_1():
    """
    Return two centers which are 2-dimensional vectors whose keys are 'mu_x' and 'mu_y'.
    Assume the initial centers are
    ({'mu_x': -2, 'mu_y': 0}, {'mu_x': 3, 'mu_y': 0})
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    
    return {'mu_x': -0.5, 'mu_y': 1.5}, {'mu_x': 3, 'mu_y': 1.5}
    # END_YOUR_ANSWER

def problem_3a_2():
    """
    Return two centers which are 2-dimensional vectors whose keys are 'mu_x' and 'mu_y'.
    Assume the initial centers are
    ({'mu_x': -1, 'mu_y': -1}, {'mu_x': 2, 'mu_y': 3})
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return {'mu_x': -1, 'mu_y': 0}, {'mu_x': 2, 'mu_y': 2}
    # END_YOUR_ANSWER

############################################################
# Problem 3: k-means implementation
############################################################

def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run for (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments, (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_ANSWER (our solution is 40 lines of code, but don't worry if you deviate from this)
    
    # initialize centroids randomly
    centroids = random.sample(examples, K)

    # calculate squared of each value (centroids, examples)
    centroids_squared = [dotProduct(i, i) for i in centroids]
    examples_squared = [dotProduct(i, i) for i in examples]

    def distance(idx_k, idx_e):
        # (a-b)^2 = a^2 + b^2 -2*a*b
        return centroids_squared[idx_k] + examples_squared[idx_e] - 2 * dotProduct(centroids[idx_k], examples[idx_e])

    # initialize assignments
    assignments = []
    iters = 0
    while iters < maxIters:

        # choose best assignments given centroids and update the assignments
        update_assign = []
        for i in range(len(examples)):
            # create the list of distances between every k points and examples[i] data point
            min_k = 0
            min_dist = distance(0,i)
            for j in range(K):
                if(min_dist > distance(j,i)):
                    min_dist = distance(j,i)
                    min_k = j
            # find the k value that has minimum distance for the examples[i] and update to update_assign
            update_assign.append(min_k)
        if update_assign == assignments: # if no change -> escape from loop
            break 
        assignments = update_assign # update the assignments

        # choose best centroids given assignments (calculate the average point)
        k_pos_list = {}
        k_size_list = {}
        update_centroid = {}
        for i in range(K):
            k_pos_list[i] = {} # for each k clusters, store the coordinate of centroids
            k_size_list[i] = 0 # for each k clusters, store the number of assignments of examples
            update_centroid[i] = {}
        # first, search all assignments of examples and set k_pos_list, k_size_list
        for i in range(len(examples)):
            # sum the position points of examples those assignment is k
            increment(k_pos_list[assignments[i]], 1, examples[i])
            # calculate the number of examples those assignment is k
            k_size_list[assignments[i]] += 1
        # second, search all k cluster, calculate the average points for new centroids
        for i in range(K):
            # if there are examples those assignmet is k, calculate the average
            if k_size_list[i] > 0:
                for dim in range(len(k_pos_list[i])):
                    sum_cord_dim = k_pos_list[i][dim]
                    update_centroid[i][dim] =  sum_cord_dim / k_size_list[i]
            centroids[i] = update_centroid[i]
            centroids_squared[i] = dotProduct(centroids[i], centroids[i])

        # update for total loop
        iters += 1
    
    # calculate total loss
    loss = 0
    for i in range(len(examples)):
        loss += distance(assignments[i],i)
    #loss = sum(distance(assignments[i], i) for i in range(len(examples)))

    return (centroids, assignments, loss)

    # END_YOUR_ANSWER


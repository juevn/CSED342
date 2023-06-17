import collections
import math

############################################################
# Problem 1a
def denseVectorDotProduct(v1, v2):    
    """
    Given two dense vectors |v1| and |v2|, each represented as list,
    return their dot product.
    You might find it useful to use sum(), and zip() and a list comprehension.
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    # raise NotImplementedError
    if len(v1) != len(v2):
        return 0

    return sum(i[0] * i[1] for i in zip(v1, v2))
    # END_YOUR_ANSWER

############################################################
# Problem 1b
def incrementDenseVector(v1, scale, v2):
    """
    Given two dense vectors |v1| and |v2| and float scalar value scale, return v = v1 + scale * v2.
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    # raise NotImplementedError
    sum_v = []
    for i in range(len(v1)):
        sum_v.append(v1[i] + scale * v2[i])
    return sum_v
    # END_YOUR_ANSWER

############################################################
# Problem 1c
def dense2sparseVector(v):
    """
    Given a dense vector |v|, return its sparse vector form,
    represented as collection.defaultdict(float).
    
    For exapmle:
    >>> dv = [0, 0, 1, 0, 3]
    >>> dense2sparseVector(dv)
    # defaultdict(<class 'float'>, {2: 1, 4: 3})
    
    You might find it useful to use enumerate().
    """
    # raise NotImplementedError
    from collections import defaultdict
    counter = defaultdict(float)
    for num in range(len(v)):
        if(v[num] != 0):
            counter[num] = v[num]
    return counter


############################################################
# Problem 1d
def sparseVectorDotProduct(v1, v2):  # -> sparse vector product, dense vectoer product, dense sparse matmul
    """
    Given two sparse vectors |v1| and |v2|, each represented as collection.defaultdict(float),
    return their dot product.
    You might find it useful to use sum() and a list comprehension.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    # raise NotImplementedError
    sum = 0
    for num in v1:
        sum += v1[num] * v2.get(num,0)
        
    return sum
    # END_YOUR_ANSWER

############################################################
# Problem 1e
def incrementSparseVector(v1, scale, v2):
    """
    Given two sparse vectors |v1| and |v2|, return v = v1 + scale * v2.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_ANSWER (our solution is 4 lines of code, but don't worry if you deviate from this)
    # raise NotImplementedError
    from collections import defaultdict
    v = defaultdict(float)
    for i in v2:
        v[i] = v1[i] + scale * v2.get(i,0)
    return v
    # END_YOUR_ANSWER

############################################################
# Problem 2a
def minkowskiDistance(loc1, loc2, p = math.inf): 
    """
    Return the Minkowski distance for p between two locations,
    where the locations are n-dimensional tuples.
    the Minkowski distance is generalization of
    the Euclidean distance and the Manhattan distance. 
    In the limiting case of p -> infinity,
    the Chebyshev distance is obtained.
    
    For exapmle:
    >>> p = 1 # manhattan distance case
    >>> loc1 = (2, 4, 5)
    >>> loc2 = (-1, 3, 6)
    >>> minkowskiDistance(loc1, loc2, p)
    # 5

    >>> p = 2 # euclidean distance case
    >>> loc1 = (4, 4, 11)
    >>> loc2 = (1, -2, 5)
    >>> minkowskiDistance = (loc1, loc2)  # 9

    >>> p = math.inf # chebyshev distance case
    >>> loc1 = (1, 2, 3, 1)
    >>> loc2 = (10, -12, 12, 2)
    >>> minkowskiDistance = (loc1, loc2, math.inf)
    # 14
    
    """
    # BEGIN_YOUR_ANSWER (our solution is 4 lines of code, but don't worry if you deviate from this)
    # raise NotImplementedError
    if(p == math.inf):
        return max(abs(e1-e2) for e1, e2 in zip(loc1,loc2))

    return sum(abs(e1-e2)**p for e1, e2 in zip(loc1,loc2))**(1/p)

    # END_YOUR_ANSWER


############################################################
# Problem 2b
def getLongestWord(text):
    """
    Given a string |text|, return the longest word in |text|. 
    If there are ties, choose the word that comes first in the alphabet.
    
    For example:
    >>> text = "tiger cat dog horse panda"
    >>> getLongestWord(text) # 'horse'
    
    Note:
    - Assume there is no punctuation and no capital letters.
    
    Hint:
    - max/min function returns the maximum/minimum item with respect to the key argument.
    """

    # BEGIN_YOUR_ANSWER (our solution is 4 line of code, but don't worry if you deviate from this)
    # raise NotImplementedError
    wordlist = text.split()
    newlist = []
    maxsize = -1
    for word in wordlist:
        if(len(word)>maxsize):
            maxsize = len(word)

    for i in range(len(wordlist)):
        if (len(wordlist[i]) == maxsize):
            newlist.append(wordlist[i])

    newlist.sort()

    return newlist[0]

    # END_YOUR_ANSWER

############################################################
# Problem 2c
def getFrequentWords(text, freq):
    """
    Splits the string |text| by whitespace
    and returns a set of words that appear at a given frequency |freq|.
    """
    # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
   
    wordlist = text.split(' ')
    word_counter = dict()
    for word in wordlist:
        if(word not in word_counter):
            word_counter[word] = 1
        else:
            word_counter[word] += 1
    
    wordset = set()
    for word in word_counter:
        if(word_counter[word]==freq):
            wordset.add(word)

    return wordset
    # END_YOUR_ANSWER 
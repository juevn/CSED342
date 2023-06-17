#!/usr/bin/env python

import graderUtil, collections, random

grader = graderUtil.Grader()
submission = grader.load('submission')

try:
    import solution
    grader.addHiddenPart = grader.addBasicPart
    SEED = solution.SEED
    solution_exist = True
except ModuleNotFoundError:
    SEED = 42
    solution_exist = False

############################################################
# Problem 1a: denseVectorDotProduct
grader.addBasicPart('1a-0-basic', lambda : grader.requireIsEqual(11, submission.denseVectorDotProduct([1,2],[3,4])), 1)

def randDenseVec():
    v = random.sample(range(-10,10),10)
    return v
def test():
    random.seed(SEED)
    def get_gen():
        for _ in range(10):
            v1 = randDenseVec()
            v2 = randDenseVec()
            pred = submission.denseVectorDotProduct(v1, v2)
            
            if solution_exist:
                answer = solution.denseVectorDotProduct(v1, v2)
                yield grader.requireIsTrue(pred == answer)
            else:
                yield True
    all(get_gen())

grader.addHiddenPart('1a-1-hidden', test, 1)

############################################################
# Problem 1b: incrementDenseVector
grader.addBasicPart('1b-0-basic', lambda : grader.requireIsEqual([13,17], submission.incrementDenseVector([1,2],3,[4,5])), 1)

def test():
    random.seed(SEED)
    def get_gen():
        for _ in range(10):
            v1 = randDenseVec()
            v2 = randDenseVec()
            s = random.randint(-10, 10)/2
            pred = submission.incrementDenseVector(v1, s, v2)
            if solution_exist:
                answer = solution.incrementDenseVector(v1, s, v2)
                yield grader.requireIsTrue(pred==answer)
            else:
                yield True
    all(get_gen())

grader.addHiddenPart('1b-1-hidden', test, 1)

############################################################
# Problem 1c: dense2sparseVector
grader.addBasicPart('1c-0-basic', lambda : grader.requireIsEqual(collections.defaultdict(float, {0:1, 3:2}), submission.dense2sparseVector([1,0,0,2])), 1)

def test():
    random.seed(SEED)
    def get_gen():
        for _ in range(10):
            v = randDenseVec()
            pred = submission.dense2sparseVector(v)
            if solution_exist:
                answer = solution.dense2sparseVector(v)
                yield grader.requireIsTrue(pred==answer)
            else:
                yield True
    all(get_gen())

grader.addHiddenPart('1c-1-hidden', test, 1)

############################################################
# Problem 1d: sparseVectorDotProduct
def test():
    v1 = collections.defaultdict(float, {'a': 5})
    v2 = collections.defaultdict(float, {'b': 2, 'a': 3})
    grader.requireIsEqual(15, submission.sparseVectorDotProduct(v1, v2))

grader.addBasicPart('1d-0-basic', test, 1)

def randSparseVec():
    v = collections.defaultdict(float)
    for _ in range(10):
        v[random.randint(0, 10)] = random.randint(0, 10) - 5
    return v

def test():
    random.seed(SEED)
    def get_gen():
        for _ in range(10):
            v1 = randSparseVec()
            v2 = randSparseVec()
            pred = submission.sparseVectorDotProduct(v1, v2)
            if solution_exist:
                answer = solution.sparseVectorDotProduct(v1, v2)
                yield grader.requireIsTrue(pred == answer)
            else:
                yield True
    all(get_gen())

grader.addHiddenPart('1d-1-hidden', test, 1)

############################################################
# Problem 1e: incrementSparseVector

def veceq(vec1, vec2):
    def veclen(vec):
        return sum(1 for k, v in vec.items() if v != 0)
    if veclen(vec1) != veclen(vec2):
        return False
    else:
        return all(v == vec2.get(k, 0) for k, v in vec1.items())

def test():
    v1 = collections.defaultdict(float, {'a': 1})
    v2 = collections.defaultdict(float, {'b': 2, 'a': 3})
    pred = submission.incrementSparseVector(v1, 2, v2)
    grader.requireIsEqual(collections.defaultdict(float, {'a': 7, 'b': 4}), pred)
grader.addBasicPart('1e-0-basic', test, 1)

def test():
    random.seed(SEED)
    def get_gen():
        for _ in range(10):
            v1 = randSparseVec()
            v2 = randSparseVec()
            s = random.randint(-10, 10)/2
            pred = submission.incrementSparseVector(v1, s, v2)
            if solution_exist:
                answer = solution.incrementSparseVector(v1, s, v2)
                yield grader.requireIsTrue(veceq(pred, answer))
            else:
                yield True
    all(get_gen())
grader.addHiddenPart('1e-1-hidden', test, 1)

############################################################
# Problem 2a: minkowskiDistance

def test():
    grader.requireIsEqual(3, submission.minkowskiDistance((0, 1), (2, 4)))
    grader.requireIsEqual(4, submission.minkowskiDistance((0, 1), (2, 3), p=1))
    grader.requireIsEqual(5, submission.minkowskiDistance((0, 1), (3, 5), p=2))

grader.addBasicPart('2a-0-basic', test, 1)

def test():
    def get_gen():
        for _ in range(100):
            x = tuple(random.randint(0, 10) for _ in range(10))
            y = tuple(random.randint(0, 10) for _ in range(10))
            p = random.randint(1, 11) / 2
            if p <= 5:
                pred = submission.minkowskiDistance(x, y, p)
            elif p > 5:
                pred = submission.minkowskiDistance(x, y)

            if solution_exist:
                if p <= 5:
                    answer = solution.minkowskiDistance(x, y, p)
                elif p > 5:
                    answer = solution.minkowskiDistance(x, y)
                yield grader.requireIsTrue(pred == answer)
            else:
                yield True
    all(get_gen())
grader.addHiddenPart('2a-1-hidden', test, 1)

############################################################
# Problem 2b: getLongestWord
def test():
    grader.requireIsEqual('longest', submission.getLongestWord('which is the longest word'))
    grader.requireIsEqual('cat', submission.getLongestWord('sun cat dog'))
    grader.requireIsEqual('10000', submission.getLongestWord(' '.join(str(x) for x in range(100000))))

grader.addBasicPart('2b-0-basic', test, 1)

def test():
    chars = tuple(map(lambda x: chr(x), range(ord('a'), ord('z') + 1)))
    def get_gen():
        for _ in range(20):
            text = ' '.join(''.join(random.choice(chars) for _ in range(random.choice(range(5, 25))))
                            for _ in range(500))
            pred = submission.getLongestWord(text)
            if solution_exist:
                answer = solution.getLongestWord(text)
                yield grader.requireIsTrue(pred == answer)
            else:
                yield True
    all(get_gen())
grader.addHiddenPart('2b-1-hidden', test, 1)

############################################################
# Problem 2c: getFrequentWords

def test():
    grader.requireIsEqual({'the', 'fox'}, submission.getFrequentWords('the quick brown fox jumps over the lazy fox',2))
grader.addBasicPart('2c-0-basic', test, 1)

def test(numTokens, numTypes):
    def get_gen():
        text = ' '.join(str(random.randint(0, numTypes)) for _ in range(numTokens))
        for i in range(20):
            freq = 90 + i
            pred = submission.getFrequentWords(text,freq)
            if solution_exist:
                answer = solution.getFrequentWords(text,freq)
                yield grader.requireIsTrue(pred == answer)
            else:
                yield True
    all(get_gen())
grader.addHiddenPart('2c-1-hidden', lambda : test(10000, 100), 1)

grader.grade()
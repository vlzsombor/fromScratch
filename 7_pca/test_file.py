import numpy as np
import pytest


aGlobal = np.array([[1,0,3], [4,5,6], [7,0,9]])


def test_covariance_vector():
    a1 = np.array([1,2,3])
    a2 = np.array([4,5,12])

    covSample = np.cov(a1, a2, rowvar=False)
    covPop = np.cov(a1, a2, ddof=0,rowvar=False)
    varSample = np.var(a1, ddof=1)
    varPop = np.var(a1)

    assert covSample[0][0] == varSample

    assert np.cov(a1, a2, ddof=1)[0][0] == varSample
    assert np.cov(a1, a2, ddof=1)[0][0] == covSample[0][0]
    assert covPop[0][0] == pytest.approx(varPop, abs = 0.01)
    assert covPop[0][0]/(np.std(a1)**2) == pytest.approx(1, abs = 0.01)
    assert covPop[0][0]/(np.var(a1, ddof=0)) == pytest.approx(1, abs = 0.01)

def test_covariance_one_vector():
    a1 = np.array([170, 165, 180, 1,3])
    covSample = np.cov(a1)
    assert covSample.shape == ()

def test_covariance_matrix():
    a1 = np.array([[1,0,3], [4,5,6], [7,0,9]])

    covSample = np.cov(a1)
    covPop = np.cov(a1, ddof=0)
    varSample = np.var(a1, ddof=1)
    varPop = np.var(a1)

    assert covSample[0][0] == varSample

    assert np.cov(a1, ddof=1)[0][0] == varSample
    assert np.cov(a1, ddof=1)[0][0] == covSample[0][0]
    assert covPop[0][0] == pytest.approx(varPop, abs = 0.01)
    assert covPop[0][0]/(np.std(a1)**2) == pytest.approx(1, abs = 0.01)
    assert covPop[0][0]/(np.var(a1, ddof=0)) == pytest.approx(1, abs = 0.01)


def test_eigenValues():

    eigenvalues, eigenvectors = np.linalg.eigh(aGlobal)

    assert True


def test_Var():
    a = [1,2,3,4,5,6]
    var = np.var(a)
    assert var == pytest.approx(2.91666, abs=0.01)



from numpy import array, diag
from numpy.linalg import eig, inv

def test_a():

    a = np.array([[1,0,3], 
                  [4,5,6], 
                  [7,0,9]])




    eigenvalues, eigenvectors = np.linalg.eigh(a)
    eigenvalues2, eigenvectors2 = np.linalg.eig(a)

    print(eigenvectors)
    print(eigenvectors2)

    assert eigenvectors[0][0]**2 + eigenvectors[0][1]**2 + eigenvectors[0][2]**2 == pytest.approx(1, abs=0.001)

def test_b():

    a = np.array([[1,0,3], [4,5,6], [7,0,9]])
    b = a[:,::-1]

    print(b)

def test_test():
    A = array([
        [1, 2],
        [4, 5]
    ])
    eigenvals, eigenvecs = eig(A)
    print("EIGENVALUES")
    print(eigenvals)
    print("\nEIGENVECTORS")
    print(eigenvecs)
    print("\nREBUILD MATRIX")
    Q = eigenvecs
    R = inv(Q)
    L = diag(eigenvals)
    B = Q @ L @ R
    print(B)
    """
    EIGENVALUES
    [-0.46410162 6.46410162]
    EIGENVECTORS
    [[-0.80689822 -0.34372377]
    [ 0.59069049 -0.9390708 ]]
    REBUILD MATRIX
    [[1. 2.]
    [4. 5.]]
    """
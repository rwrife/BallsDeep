
import numpy


def softmax(w):
    """Calculate the softmax of a list of numbers w.

    Parameters
    ----------
    w : list of numbers

    Return
    ------
    a list of the same length as w of non-negative numbers

    Examples
    --------
    >>> softmax([0.1, 0.2])
    array([ 0.47502081,  0.52497919])
    >>> softmax([-0.1, 0.2])
    array([ 0.42555748,  0.57444252])
    >>> softmax([0.9, -10])
    array([  9.99981542e-01,   1.84578933e-05])
    >>> softmax([0, 10])
    array([  4.53978687e-05,   9.99954602e-01])
    """
    e = numpy.exp(numpy.array(w))
    print e
    print numpy.sum(e)
    dist = e / numpy.sum(e)
    return dist


print softmax([0.1,0.2])
from typing import List


def find_sum(a, b):
    """Given integers a and b where a < b, return the sum a + (a + 1) + (a + 2) ... + b"""
    c = a
    i = 0
    total = 0
    for i in range(b - a + 1):
        total += (c + i)

    return total
    pass


def get_info(name, age, ssn):
    """
    Given a name, age, and ssn, this returns a string of the form, complete with newlines and tabs.

    Example: get_info('Scott', 100, 1234567890) returns:
    Name: Scott
        Age: 100
        SSN: 1234567890
    get_info('Scott', 'older', '123-45-7890') returns:
    Name: Scott
        Age: older
        SSN: 123-45-7890
    """
    tobereturned = f'''Name: {name} \n  Age: {age} \n   SSN: {ssn}'''
    return tobereturned
    pass


def get_method_and_var_names(obj):
    """
    Given an object, this method returns a list of the names of the methods which do not contain "__".
    Example:
    get_method_and_var_names("a string") returns the list:
    ['capitalize', 'casefold', 'center', 'count', 'encode', 'endswith', 'expandtabs', 'find', 'format', 'format_map', 'index', 'isalnum', 'isalpha', 'isdecimal', 'isdigit', 'isidentifier', 'islower', 'isnumeric', 'isprintable', 'isspace', 'istitle', 'isupper', 'join', 'ljust', 'lower', 'lstrip', 'maketrans', 'partition', 'replace', 'rfind', 'rindex', 'rjust', 'rpartition', 'rsplit', 'rstrip', 'split', 'splitlines', 'startswith', 'strip', 'swapcase', 'title', 'translate', 'upper', 'zfill']

    import numpy as np; get_method_and_var_names(np) returns the list:
    ['ALLOW_THREADS', 'AxisError', 'BUFSIZE', 'CLIP', 'ComplexWarning', ... 'where', 'who', 'zeros', 'zeros_like']
    (The full list is considerably longer than that)
    """
    myList = []
    list = dir(obj)
    for s in list:
        if "__" not in s:
            myList.append(s)

    return myList
    pass


def evaluate(f, x):
    """
    Given a function f, and a value x, this method returns the result of calling f on x.

    Example:
    evaluate(len, [1, 2, 3])
    returns 3

    def foo(var):
        return var + 5
    evaluate(foo, 1)
    returns 6.
    """
    y = f(x)
    return y
    pass




def threshold_factory(thresh):
    """
    This method returns a function.
    The function it returns is a threshold function which takes an argument x and returns:
    True if x < thresh
    False if x >= thresh

    Example:
    t5 = threshold_factory(5)
    t5(1) # returns True
    t5(6) # returns False
    t_other = threshold_factory(1.23)
    t_other(1.234) # returns False
    """

    def returnfunction(arg):

        return arg < thresh


    return returnfunction


t = threshold_factory(10)
print(get_info('Scott', 20, 1234))
print(t(1))
print(t(2))
print(t(3))
print(t(10))
print(t(15))
print(t(100))

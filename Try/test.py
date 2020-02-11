def foo():
    print("hello world in foo")
    bar()

def bar():
    print("hello world in bar")





import numpy as np
arr1 = np.arange(0,30,5).reshape(2,3)
it = np.nditer(arr1, flags=['multi_index'],op_flags=['readwrite'])
while not it.finished:
    print(it.multi_index)
    it.iternext()
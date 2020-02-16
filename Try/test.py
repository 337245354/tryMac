def foo():
    print("hello world in foo")
    bar()

def bar():
    print("hello world in bar")





import numpy as np
arr1 = np.arange(0,30,2).reshape(-1,3)
it = np.nditer(arr1, flags=['multi_index'],op_flags=['readwrite'])
print(arr1)
# while not it.finished:
#     print(it.multi_index)
#     it.iternext()

    # (0, 0)
    # (0, 1)
    # (0, 2)
    # (1, 0)
    # (1, 1)
    # (1, 2)

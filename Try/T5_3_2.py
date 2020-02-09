import numpy as np
a = np.array([[-1,1,2,3],
            [-2,-1,4,5]])
a1 = np.array([-1,1,2,3])
def _sigmoid(in_data):
    return 1 / (1 + np.exp(-in_data))
print(_sigmoid(a1))
print(_sigmoid(a))



import numpy as np
a = np.array([[-1,1,2,3],
        [-2,-1,4,5]])
a1 = np.array([-1,1,2,3])
def _relu(in_data):
    return np.maximum(0,in_data)
print(_relu(a))
print(_relu(a1))

import numpy as np
arr1 = np.array([[0,0,0],[1,1,1],[2,2,2],[3,3,3]])
arr2 = np.array([1,2,3])
arr_sum = arr1 + arr2
print(arr1.shape)
print(arr2.shape)
print(arr_sum)


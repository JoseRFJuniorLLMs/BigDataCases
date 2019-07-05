import numpy as np

a1 = np.array([1,2,3,4,5])
print(a1)
type(a1)

a2 = np.array([[1,2,3,4,5],[1,2,7,8,9]])
a3 = np.array([[6,8,9,4,5],[9,5,7,8,9]])

a3.shape

a4 = a2.dot(a3.reshape(5,2))
a4

a5 = np.sin(a4)
a6 = np.cos(a4)

a7 = a5**2 + a6**2

np1 = np.zeros((4,5),dtype = int)
np1 = np.ones(10,dtype = int)


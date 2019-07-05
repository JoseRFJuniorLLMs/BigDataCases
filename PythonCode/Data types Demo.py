
a = 101


print(a)
type(a)

a = a +6
a = a - 890


a = a/12

flag = bool()

flag = True

#String
a= "my name is XYZ"

a.title()

b = a +" " + "and I am Cool"

#Int, Float, String, Bool

#List, Tuple, Set and Dictionary

list1 = [1,2,3,4,5]
list2 = [3,4,5,6,80]
list3 = list1 + list2

list3.append(98)
print(list3)
len(list3)


list_new = [1,1.0,"String",[1,2,3,5]]
list_new[3] = "New value"

print(list_new)

#Tuple
tuple1 = tuple(list_new)
tuple1[1] = 100

set_new = set([1,2,3,4,4,4,4,4,5,5,5,5])

dict1 = {"A":123,"B":"the"}
dict1["B"]
dict1["T"] = 100

#-------------
import numpy as np
a1=np.array([[1,2,4,5,6,8],[1,2,4,5,6,8],[1,2,4,5,6,8]])
a2=a1.cumsum()
a2 =a1.dot(a1.reshape(6,3))
a3=a1.repeat(2)
a1.resize()

np.zeros(5,dtype=int)
np.random.choice([1,2,4,5,6,7])
np.sin([1,2,3]) **2 +np.cos([1,2,3]) **2 
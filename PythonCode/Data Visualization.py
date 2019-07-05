import matplotlib.pyplot as plt
import numpy as np

x = np.random.randint(100,500,50)
y = np.random.randint(100,1000,50)
y1 = np.random.randint(100,1000,50)


plt.scatter(x,y,marker = "s",color="red")
plt.scatter(x,y1,marker = "^",color="blue")
plt.title("This is a sample scatter plot")
plt.xlabel("X axis label")
plt.ylabel("Y axis label")
plt.show()

x = np.arange(5)
y = np.random.randint(50,100,5)
y1 = np.random.randint(50,100,5)
#xlist= list("ABCDE")
xlist= ["English","Math","History","Science","Geography"]
plt.xticks(x,xlist)
plt.bar(x,y,color="red")
plt.bar(x,y1,color="blue",bottom=y)
plt.title("Marks scored by John")
plt.xlabel("Subjects")
plt.ylabel("Marks obtained")
plt.show()

x = np.arange(1,7,1)
y = np.random.randint(50,100,6)
xlist = ["Sem1", "Sem2", "Sem3", "Sem4","Sem5", "Sem6"]
plt.xticks(x,xlist)
plt.title("Overall College performance")
plt.ylabel("Percentage")
plt.xlabel("Semesters")
plt.plot(x,y,"b^")
plt.plot(x,y,"red")

df.columns

plt.title("Overall dsitribution of employee performance")
plt.ylabel("Number of employees")
plt.xlabel("Satisfaction level")
plt.hist(df["satisfaction_level"],bins=10)
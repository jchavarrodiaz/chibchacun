import numpy as np
import matplotlib.pyplot as plt

lowerBound = 0.25
upperBound = 0.75
myMatrix = np.random.rand(100,100)

myMatrix =np.ma.masked_where((lowerBound < myMatrix) &
                             (myMatrix < upperBound), myMatrix)


fig,axs=plt.subplots(2,1)
#Plot without mask
axs[0].imshow(myMatrix.data)

#Default is to apply mask
axs[1].imshow(myMatrix)

plt.show()
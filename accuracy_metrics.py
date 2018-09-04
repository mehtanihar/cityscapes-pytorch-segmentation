import numpy as np
from PIL import Image


def pixel_accuracy(image1,image2):
	image1=np.array(image1)	
	image2=np.array(image2)
	[row,col]=image1.shape
	image1=np.reshape(image1,(row*col,1))
	image2=np.reshape(image2,(row*col,1))
	count=0
	total_count=0
	for i in range(row*col):
			total_count+=1
			if(image1[i]==image2[i]):
				count+=1
		
	return count/(total_count+1e-8)

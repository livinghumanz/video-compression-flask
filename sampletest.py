# Program To Read video 
# and Extract Frames 
import cv2 
from matplotlib.image import imread
import matplotlib.pyplot as plt
#----------coloured----
from PIL import Image
import numpy
#----------------------
import os

# open the image and return 3 matrices, each corresponding to one channel (R, G and B channels)
def openImage(imOrig):
    #imOrig = Image.open(imagePath)
    im = numpy.array(imOrig)

    aRed = im[:, :, 0]
    aGreen = im[:, :, 1]
    aBlue = im[:, :, 2]

    return [aRed, aGreen, aBlue, imOrig]

# compress the matrix of a single channel
def compressSingleChannel(channelDataMatrix, singularValuesLimit):
    uChannel, sChannel, vhChannel = numpy.linalg.svd(channelDataMatrix)
    aChannelCompressed = numpy.zeros((channelDataMatrix.shape[0], channelDataMatrix.shape[1]))
    k = singularValuesLimit

    leftSide = numpy.matmul(uChannel[:, 0:k], numpy.diag(sChannel)[0:k, 0:k])
    aChannelCompressedInner = numpy.matmul(leftSide, vhChannel[0:k, :])
    aChannelCompressed = aChannelCompressedInner.astype('uint8')
    return aChannelCompressed

  
# Function to extract frames 
def FrameCapture(path): 
      
    # Path to video file 
    vidObj = cv2.VideoCapture(path) 
    #get fps  and detail of the video
    fps=vidObj.get(cv2.CAP_PROP_FPS)
    print("fps= ",fps)
    height,width=vidObj.get(4),vidObj.get(3)
    print(height,width)
  
    count = 0
  
    # checks whether frames were extracted 
    ret, image = vidObj.read() 
    images=[]
    plt.rcParams['figure.figsize']=[16,8]
    while ret:
        
        #------SVD compression---
        # get chanel matricesof the image (RGB)
        aRed, aGreen, aBlue, originalImage = openImage(image)
        print(aRed,aGreen,aBlue)
        cv2.imshow(Image.fromarray(aBlue))
        
        images.append(image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
	        break
        ret, image = vidObj.read() 

        '''x=np.mean(image,-1) # converting to gray scale
        #img=plt.imshow(x)
        #plt.imshow(x)
        #img.set_cmap('gray')
        #plt.axis('off')
        #plt.show()

        U, S, VT = np.linalg.svd(x,full_matrices=False)
        S=np.diag(S)
        j=2
        r=150
        #construct approximate image
        xapprox = np.dot(U[:,:r],np.dot(S[0:r,:r],(VT[:r,:])))
        plt.figure(j+1)
        #j+=1
        #plt.imshow(xapprox)
        #img=plt.imshow(xapprox)
        #img.set_cmap('gray')
        #plt.axis("off")
        #plt.title("r = " + str(r))
        #plt.show()
        



        #------Done SVD compression----
        images.append(image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
	        break
        ret, image = vidObj.read() 

    print("count: ",count,"list len :",len(images),"width->",vidObj.get(3),"height->",vidObj.get(4))

    #cv2.imshow("frame",images[510])
    #im=cv2.cvtColor(images[309],cv2.COLOR_BGR2RGB)
    #cv2.imshow("frame",im)'''

    cv2.waitKey(0)  
    vidObj.release()
    #closing all open windows  
    cv2.destroyAllWindows() 
    



# Driver Code 
if __name__ == '__main__': 
  
    # Calling the function 
    FrameCapture("./11sec_video_add1.mp4")
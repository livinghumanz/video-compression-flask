'''
Prerequisites:
pip install numpy
pip install pillow
pip install opencv
'''

import numpy
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import os

#test_image=[]
test_image_n={}
# FUNCTION DEFINTIONS:

# open the image and return 3 matrices, each corresponding to one channel (R, G and B channels)
def openImage(imOrig):
    #imOrig = Image.open(imagePath)
    im = numpy.array(imOrig)

    aRed = im[:, :, 0]
    aGreen = im[:, :, 1]
    aBlue = im[:, :, 2]

    return [aRed, aGreen, aBlue]


# compress the matrix of a single channel
def compressSingleChannel(channelDataMatrix, singularValuesLimit,count=0):
    uChannel, sChannel, vhChannel = numpy.linalg.svd(channelDataMatrix)
    aChannelCompressed = numpy.zeros((channelDataMatrix.shape[0], channelDataMatrix.shape[1]))
    k = singularValuesLimit

    leftSide = numpy.matmul(uChannel[:, 0:k], numpy.diag(sChannel)[0:k, 0:k])
    aChannelCompressedInner = numpy.matmul(leftSide, vhChannel[0:k, :])
    aChannelCompressed = aChannelCompressedInner.astype('uint8')
    if(count==9 or count==6):
        #plt.figure(1)
        plt.semilogy(numpy.diag(sChannel))
        plt.title("Losses and gain single")
        plt.savefig(os.getcwd()+"/static/DBA_Report/plot_losses.jpg")
        plt.clf()
        
    
    return aChannelCompressed


# split the video into frame for compression and return list
def frameByFrameCompression(path):
    print('*** Processing and analysing your file ...')

    vidObj = cv2.VideoCapture(path)

    #get Frame per second
    fps=(vidObj.get(cv2.CAP_PROP_FPS) or vidObj.get(5))

    # image width and height:
    #imageWidth = vidObj.get(3)
    #imageHeight = vidObj.get(4)
    size=(410,210)

    # get Codec of the video
    fourcc = vidObj.get(6)#cv2.CV_CAP_PROP_FOURCC)

    # get video object 
    ret, image = vidObj.read() 
    test_image_n["original"]=image
    generate_plot(image,"visualize_base.jpg")
    images=[]
    count=0
    print("** Generating Optimal with "+str(fps)+" fps for (w,h)->",size,fourcc)
    while ret:
        #if count>=10:
        #    break
        if count%3 == 0:
            image=cv2.resize(image,size)
            aRed, aGreen, aBlue = openImage(image)
            #plt.imshow(Image.fromarray(aRed))
            #plt.show()
            #plt.imshow(Image.fromarray(aGreen))
            #plt.show()
            #plt.imshow(Image.fromarray(aBlue))
            #plt.show()
            
            # number of singular values to use for reconstructing the compressed image
            singularValuesLimit = 210

            aRedCompressed = compressSingleChannel(aRed, singularValuesLimit)
            aGreenCompressed = compressSingleChannel(aGreen, singularValuesLimit,count)
            aBlueCompressed = compressSingleChannel(aBlue, singularValuesLimit)

            imr = Image.fromarray(aRedCompressed, mode=None)
            img = Image.fromarray(aGreenCompressed, mode=None)
            imb = Image.fromarray(aBlueCompressed, mode=None)

            newImage = Image.merge("RGB", (imr, img, imb))
            #newImage.show()
            #cv2 merge test
            newImage_cv2 = numpy.asarray(newImage)
            if count == 0:
                newImage.save(os.getcwd()+"/static/DBA_Report/knn_s.jpg")
                r, g, b = cv2.split(newImage_cv2)
                test_image_n["knn_s"]= cv2.merge([b,g,r])
            #cv2.imshow("frame",newImage_cv2)

            # show new and original image 
            #originalImage.show()
            #if count>100:
            #newImage.show()

            # Append the image and iterate for next frame
            images.append(newImage_cv2)
        if cv2.waitKey(25) & 0xFF == ord('q'):
	        break
        ret, image = vidObj.read()
        count+=1
    
    vidObj.release()
    #closing all open windows  
    cv2.destroyAllWindows()
    print("* completed processing . .")
    return [images,size,int(fps/3),fourcc]

def compressVideo(path):
    # MAIN PROGRAM:
    #filename="./11sec_video_add1.mp4"
    #filename=path
    images,size,fps,fourcc=frameByFrameCompression(path)
    print(" Preparing Your video file . . .")
    filename="/".join(path.split("/")[0:-1])+"/"+"1.".join(path.split("/")[-1].split("."))
    print(filename)
    #print(filename)
    fourcc=cv2.VideoWriter_fourcc(*"mp4v")
    #int(fourcc)
    test_image_n["frame"]=[images[1],images[2]]
    out = cv2.VideoWriter(filename,fourcc, int(fps), size)
    for i in range(len(images)):
        out.write(images[i])
    out.release()
    GenerateDataReport()
    
    return [size,fps*3]

def GenerateDataReport():
    # knn_s,frame->[f1,f2],original
    path=os.getcwd()+"/static/DBA_Report/"
    #test_image_n["knn_s"].save(path+"knn_s.jpg")
    cv2.imwrite(path+"original_image.jpg",test_image_n["original"])
    generate_plot(test_image_n["knn_s"],"visualize_knn.jpg")
    cv2.imwrite(path+"frame1.jpg",test_image_n["frame"][0])
    cv2.imwrite(path+"frame2.jpg",test_image_n["frame"][1])
    mask_image(test_image_n["frame"][1])


    
def generate_plot(img,filename):
    #img = cv2.imread('flower1.png')
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        #plt.xlim([0,256])
        plt.ylim([0,40600])
    plt.savefig(os.getcwd()+"/static/DBA_Report/"+filename)

def mask_image(image):
    height,width,chanel=image.shape
    # mask defaulting to black for 3-channel and transparent for 4-channel
    # (of course replace corners with yours)
    mask = numpy.zeros(image.shape, dtype=numpy.uint8)
    roi_corners = numpy.array([[(width/4,height/4),(width/2,height/4),(width/2,height/2),(width/4,height/2)]], dtype=numpy.int32)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    # from Masterfool: use cv2.fillConvexPoly if you know it's convex

    # apply the mask
    masked_image = cv2.bitwise_and(image, mask)

    # save the result
    cv2.imwrite(os.getcwd()+"/static/DBA_Report/image_masked.jpg", masked_image)


    
compressVideo(os.getcwd()+"/static/video/cvideo.mp4")
#fig = plt.figure()
#ax = fig.add_subplot(111, projection = '3d')
#cv2.imshow("frame",test_image[0])
#x, y, z = test_image[1][:,:,0],test_image[1][:,:,1],test_image[1][:,:,2]

#ax.scatter(x,y,z,c='b',marker='o',label='blue')
#ax.plot_wireframe(x, y, z, rstride = 1, cstride = 3)
#ax.plot_wireframe(x1, y1, z1, rstride = 1, cstride = 3,color="red")
#plt.imshow(test_image[0])
#plt.title("Wireframe Plot Example")
#plt.tight_layout()
#plt.show()


#'''#if count==2:
# CALCULATE AND DISPLAY THE COMPRESSION RATIO
#mr = imageHeight
#mc = imageWidth

#originalSize = mr * mc * 3
#compressedSize = singularValuesLimit * (1 + mr + mc) * 3

#print('original size:')
#print(originalSize)

#print('compressed size:')
#print(compressedSize)

#print('Ratio compressed size / original size:')
#ratio = compressedSize * 1.0 / originalSize
#print(ratio)

#print('Compressed image size is ' + str(round(ratio * 100, 2)) + '% of the original image ')
#print('DONE - Compressed the image! Over and out!')'''

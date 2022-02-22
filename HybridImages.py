import cv2
import numpy as np
import numpy.fft as fft
from matplotlib import pyplot as plt


def gaussian(shape, sigma):
    r = shape[0] // 2
    c = shape[1] // 2
    y, x = np.ogrid[-r:r + 1, -c:c + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    mask = h < np.finfo(h.dtype).eps * h.max()
    h[mask] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def FilterFunction(Channel,kernel,option="H"):
    imF=fft.fft2(Channel)
    kernelF=fft.fft2(kernel,s=[Channel.shape[0],Channel.shape[1]])
    if(option=="H"):
        kernelF=1-kernelF
    outF=np.multiply(imF,kernelF)
    out=fft.ifft2(outF)
    return out


def highPassFilter(image):
    image = np.double(image)
    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]

    gaussKernel = gaussian(shape=(7, 7), sigma=0.75)
    redH = FilterFunction(red, gaussKernel, option="H")
    greenH = FilterFunction(green, gaussKernel, option="H")
    blueH = FilterFunction(blue, gaussKernel, option="H")
    out = cv2.merge([np.float32(redH.real), np.float32(greenH.real), np.float32(blueH.real)])
    return out


def lowPassFilter(image):
    image = np.double(image)
    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]

    gaussKernel = gaussian(shape=(15, 15), sigma=3)
    redH = FilterFunction(red, gaussKernel, option="L")
    greenH = FilterFunction(green, gaussKernel, option="L")
    blueH = FilterFunction(blue, gaussKernel, option="L")
    out = cv2.merge([np.float32(redH.real), np.float32(greenH.real), np.float32(blueH.real)])

    return out

def hybridImageCombine(high_frequency_image, low_frequency_image):
    hybridImage = high_frequency_image + low_frequency_image
    return hybridImage

def readImage(image_path):
    image = cv2.imread(image_path)
    return image

def displayImage(image):
    image=cv2.cvtColor(np.float32(image.real), cv2.COLOR_BGR2RGB)
    plt.imshow((image).astype(np.uint8))
    plt.show()

image_path_1 = './images/angry.jpg'
image = readImage(image_path_1)
displayImage(image)

highFreqImage = highPassFilter(image)
displayImage(highFreqImage)

image_path_2 = './images/calm.jpg'
image2 = readImage(image_path_2)
displayImage(image2)

lowFreqImage = lowPassFilter(image2)
displayImage(lowFreqImage)

hybridImage =  hybridImageCombine(highFreqImage, lowFreqImage)
displayImage(hybridImage)
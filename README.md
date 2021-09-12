# 3D Vision 

## Dealing with KITTI dataset
- the KITTI rig has a baseline of 54cm.
- to convert our stereo predictions to real-world scale, we multiply our depths by `STEREO_SCALE_FACTOR = 5.4`

## Dealing with various image formats

* PIL: RGB image [W,H,C]
* cv2: BGR image [H,W,C] returns numpy.array
* numpy: BGR image [H,W,C]
* pytorch: RGB image [B,C,H,W]

## How to Save NumPy Arrays as Image Files

* pass `ndarray` to `Image.fromarray()` which returns a `PIL.Image`. It can be saved as an image file with `save()` method.
* If a grayscale image or 2D array is passed to `Image.fromarray()`, `mode` automatically becomes `L`(grayscale). It can be saved with `save()`.

* If the datatype `dtype` of `ndarray` is `float`, an error will occur, so it is necessary to convert to `uint8`.

* If the pixel value is represented by `0.0` to `1.0`, it is necessary to multiply by `255` and convert to `uint8` and save. 

## Image Resolution 
Image resolution refers to the total number of pixels in an image along with how much detail the image portrays.

### Spatial Resolution 

Spatial resolution refers to the amount of smallest discernable details in an image known as pixel. It is indicated as 1024x1024 which means 1024 pixels on both width and height of image. Higher the spatial resolution, higher the image quality and higher the amount of pixels required to represent the image.

### Intensity Level Resolution
Intensity level resolution refers to the number of intensity level used to represent the image. THe more intensity levels used, the finer the level of discernable in an image. Intensity level resolution is usually given in terms of the number of bits used to store each intensity level. 

| number of bits | number of intensity levels |
|:--------------:|:--------------------------:|
|        1       |              2             |
|        2       |              4             |
|        4       |             16             |
|        8       |             256            |
|       16       |            65536           |

Higher the intensity level, higher the image quality and higher the size of image. Like 1920x1080px images with higher intensity level is more better than the image with lower intensity level.

### Mathematical Way to Think About Intensity and Contrast
- Intensity is the mean value, while Contrast is the standard deviation.

### Convolutional Neural Networks
- A large stride allows to shrink the size of the output
- We consider the filter $K$ to be squared and to have a odd dimension denoted by $f$ which allows each pixel to be centered in the filter and thus consider all the elements around it. The filter(kernel) $K$ must have the same number of channels as the image, so that we can apply a different filter to each channel. 
- The dimension of the image and filter(kernel) is defined as follows:
$$
dim(image) = (C,H,W)
dim(filter) = (C, f, f)
$$
- If $f=1$, it might be useful in some cases to shrink the number of channels $(C)$ without changing the other dimension $(H,W)$.
- in CNN, the $(C,f,f)$ filter's parameters are learned through backpropagation. 
- Pooling downsamples the image's features through summing up the information
- Pooling is carried out through each channel and thus it only affects the dimension $(H,W)$ and keeps $(C)$ intact.
- Pooling has no parameters to learn.
- The learned paramters at the $l^{th}$ layer are:
 * Filters with $(C \times f \times f) \times C$ parameters 
 * Bias with $(1 \times 1 \times 1) \times C$ parameters (broadcasting)
- The input to a fully-connected layer would be the result of a convolution or a pooling layer with the dimension $(C,H,W)$.
- In order to be able to plug it into the fully-connected layer, we flatten the tensor to a $1D$ vector having the dimension $(C \times H \times W,)$.
- The learned parameters at the fully-connected layer are the weights and the biases. 
- The main idea of CNN is to decrease $H$ and $W$ and increase $C$ when going deeper through the network. 
- The parameter sharing of a feature detector and the sparsity of connections (each output value depends only on a small number of inputs) is what makes CNN work so efficiently. 


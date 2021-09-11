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

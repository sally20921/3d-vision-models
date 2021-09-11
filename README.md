# 3D Vision 

## Dealing with KITTI dataset
- the KITTI rig has a baseline of 54cm.
- to convert our stereo predictions to real-world scale, we multiply our depths by `STEREO_SCALE_FACTOR = 5.4`

## Dealing with various image formats

* PIL: RGB image [W,H,C]
* cv2: BGR image [H,W,C] returns numpy.array
* numpy: BGR image [H,W,C]
* pytorch: RGB image [B,C,H,W]


# 3D Vision 

## Dealing with KITTI dataset
- the KITTI rig has a baseline of 54cm.
- to convert our stereo predictions to real-world scale, we multiply our depths by `STEREO_SCALE_FACTOR = 5.4` 

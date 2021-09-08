# if you want the model to learn the depth to metric scale, you will have to use the IMU/GPS data located in the oxts folder to replace the PoseCNN with the ground truth camera-to-camera transformation. 
# if you want to create a point cloud from a depthmap you need to backproject points into 3D using hte camera matrix. 
# The network was only trained with KITTI data, so it is unclear what it will produce when used on images taken with very different intrinsics. 

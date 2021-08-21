import os, sys
import numpy as np
from plyfile import PlyData, PlyElement
import mathplotlib.pyplot as pyplot
import glob
import cv2
import open3d as o3d

def write_ply_rgb(points, colors, filename):
    '''
    (N,3) points with RGB colors (N,3) within range [0,255] as .ply file
    '''
    colors = colors.astype(int)
    N = points.shape[0]
    with open(filename, 'w') as f:
        for i in range(N):
            c = colors[i,:]
            f.write('v %f %f %f %d %d %d\n' % (points[i,0], points[i,1], points[i,2], c[0], c[1], c[2]))

def main():
    scenelist = glob.glob('/home/data/scannet/scans/*')
    nump = 50000
    data_list = []
    img_list = []
    rgbd_list = []
    pcs_list = []
    # note here that img_list, rgbd_list is not sampled while pcs_list is sampled points
    for scene in scenelist:
        depth_intrinsic = np.loadtxt(scene+'/intrinsic/intrinsic_depth.txt')
        intrin_Cam = o3d.camera.PinholeCameraIntrinsic()

        depths = sorted(glob.glob(scene+'/depth/*.png'), key=lambda a: int(os.path.basename(a).split('.')[0]))
        imgs = sorted(glob.glob(scene+'/color/*.jpg'), key=lamda a: int(os.path.basename(a).split('.')[0]))

        for ind, (img, depth) in enumerate(zip(imgs, depths)):
            name = os.path.basename(img).split('.')[0]
            depth_img = cv2.imread(depth, -1)
            try:
                o3d_depth = o3d.geometry.Image(depth_img)
                rgb_img = cv2.resize(cv2.imread(img), (depth_im.shape[1], depth_im.shape[0]))
                img_list.append(rgb_img)
                o3d_rgb = o3d.geometry.Image(rgb_img)
                o3d_rgdb = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_rgb, o3d_depth, depth_scale=1000.0, depth_trunc = 1000.0, convert_rgb_to_intensity=False)
                rgbd_list.append(o3d_rgbd)
            except:
                continue

            intrin_cam.set_intrinsics(width=depth_im.shape[1], height=depth_im.shape[0], fx=intrinsic[1,1], fy=intrinsic[0,0], cx=intrinsic[1,2], cy=intrinsic[0,2])
            pts = o3d.geometry.PointCloud.create_from_rgbd_image(o3d_rgbd, intrin_cam, np.eye(4))

            # sampling points
            arr = np.array(pts.points)
            if len(arr) >= nump:
                sel_idx = np.random.choice(len(arr), nump, replace=False)
            else:
                sel_idx = np.random.choice(len(arr), nump, replace=True)
            temp = np.array(pts.points)[sel_idx]

            color_points = np.array(pts.colors)[sel_idx]
            # (N,3) => (3, N) for points and colors
            color_points[:, [0,1,2]] = color_points[:, [2,1,0]]

            pts.points = o3d.utility.Vector3dVector(temp)
            pts.colors = o3d.utility.Vector3dVector(color_points)
            data = np.concatenate([temp, color_points], axis=1)

            pcs_list.append(pts)
            return img_list, rgbd_list, pcs_list # pc here is sampled
            






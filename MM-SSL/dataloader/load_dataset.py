import logging
import os
import numpy as np
import open3d as o3d 
from torch.utils.data import Dataset
import torch.utils.data as data
from PIL import Image
from glob import glob
from transforms import SimCLRTrainDataTransform, PointcloudTrainDataTransform
from torchvision import transforms

def read_point_cloud():
    scenelist = sorted(glob('/home/data/seri/scannet/scans/*'))
    #print(scenelist)
    pc_path_list = []
    rgb_path_list = []
    d_path_list = []
    idx = 0

    while (idx < len(scenelist)):
        scene = scenelist[idx]
        #print(idx)
        #print(scene)
        scene = scene + '/pc'
        #if not os.path.exists(scene):
        #    continue
        framelist = sorted(glob(scene+'/*'))
        for fileidx in range(len(framelist)):
            #frame = scene + '/%d.ply' % fileidx
            if (fileidx % 25) == 0:
                temp = scene
                frame = temp + '/%d.ply' % fileidx
                pts = frame
                #print(pts)
                pc_path_list.append(pts)
                temp = temp.replace("pc", "color")
                frame = temp + '/%d.jpg' % fileidx
                img = frame
                rgb_path_list.append(img)
                temp = temp.replace("color", "depth")
                frame = temp + '/%d.png' % fileidx
                depth = frame
                d_path_list.append(depth)
            else:
                continue
        idx=idx+1
        #print(idx)

    #save_pc = np.array(pc_path_list)
    #print(pc_path_list)
    #np.save('/home/MM-SSL/pc_list', save_pc)
    #save_img = np.array(rgb_path_list)
    #np.save('/home/MM-SSL/rgb_list', save_img)
    #save_depth = np.array(d_path_list)
    #np.save('/home/MM-SSL/depth_list', save_depth)

    return pc_path_list, rgb_path_list, d_path_list


class SparseDataset(Dataset):
    def __init__(self, pc_path_list, rgb_path_list, d_path_list, pc_transform=None, rgb_transform=None, d_transform=None):
        self.pc_path_list = pc_path_list
        self.rgb_path_list = rgb_path_list
        self.d_path_list = d_path_list

        self.pc_transform = pc_transform
        self.rgb_transform = rgb_transform
        self.d_transform = d_transform

    def __len__(self):
        return len(self.pc_path_list)

    def __getitem__(self, idx):
        pc = o3d.io.read_point_cloud(self.pc_path_list[idx])
        img = Image.open(self.rgb_path_list[idx])
        depth = Image.open(self.d_path_list[idx])
    
        # pc to numpy array
        pts = np.array(pc.points)
        colors = np.array(pc.colors)
        data = np.concatenate([pts, colors], axis=1)
        if self.pc_transform is not None:
            pc0, pc1, _ = self.pc_transform(data)
        if self.rgb_transform is not None:
            img0, img1, _  = self.rgb_transform(img)
        if self.d_transform is not None:
            d0, d1, _ = self.d_transform(depth)
        else:
            d0 = transforms.ToTensor()(depth)
            d1 = transforms.ToTensor()(depth)

        return {'pc0': pc0, 'pc1': pc1, 'img0': img0, 'img1': img1, 'd0': d0, 'd1': d1}




def load_data():
    pc_path, rgb_path, d_path = read_point_cloud()
    dataset = SparseDataset(pc_path, rgb_path, d_path, pc_transform=PointcloudTrainDataTransform(), rgb_transform=SimCLRTrainDataTransform(), d_transform=None)
    dataloader = data.DataLoader(dataset, num_workers=4, pin_memory=True, shuffle=False, batch_size=32)
    return dataloader

if __name__=="__main__":
    read_point_cloud()



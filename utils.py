import sys
import os 
import numpy as np
import open3d as o3d

def load_kitti_scan(scan_path):
    points_path = os.path.join(scan_path)
    points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 4)
    points = points[:, :3]  # exclude luminance
    return points

def reformat_extension(name, format):
    name = name[:-4]
    return name + "." + format

def rescaling_points(xyz, scale):
    # goal: rescaling xyz into [-1, 1] with zero mean  
    # input - scale: 25 (meter)
    mean_xyz = np.mean(xyz)
    xyz = xyz - mean_xyz
    xyz = xyz / scale
    return xyz 
    
def parse_roi_points(xyz, roi):
    # xyz: n x 3 mat 
    # roi: 3 x 2 mat ([-min, max] for x, y, and z)
    
    def within_range(val, rg):
        return (rg[0] < val) and (val < rg[1])
    
    num_points = xyz.shape[0]
    
    xroi = roi[0, :]
    yroi = roi[1, :]
    zroi = roi[2, :]

    roi_points_idxes = []
    for ii in range(num_points):
        point = xyz[ii, :]
        if( within_range(point[0], xroi) 
            and within_range(point[1], yroi)
            and within_range(point[2], zroi)):
            roi_points_idxes.append(ii)
        
    return xyz[roi_points_idxes, :]        
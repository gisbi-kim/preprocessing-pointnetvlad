import sys
import os 
import numpy as np
import open3d as o3d
from utils import * 

# params 
num_pnvlad_points = 4096

# dataset setting 
scans_dir = '~/KITTI/dataset/sequences/00/velodyne' 
print("target dataset:", scans_dir)

scans_names = os.listdir(scans_dir)
scans_names.sort()

scans_paths = [os.path.join(scans_dir, scans_name) for scans_name in scans_names]

num_scans = len(scans_paths)
print(num_scans, "scans are to be processed")


# output paths (to save)
output_dir = "~/pointnetvlad/data/KITTI00/"


# MAIN
for ii, scan_path in enumerate(scans_paths):

    # bin to npy 
    scan = load_kitti_scan(scan_path)
    scan_name = scans_names[ii]
    print("load scan", scan_name, ":", scan.shape)

    ### processing ###
    # downsampling 
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scan)
    pcd = pcd.voxel_down_sample(voxel_size=0.1)
    print("pre-downsampling with 10cm cubic (to save time):", pcd)

    # parse ROI points: [-25, 25]m cubic rather than [-80, 80]m cubic
    # - note: at RAL19 said, We empirically checked that an increased window size ([-80 m, 80 m]) deteriorated their performance for PointNetVLAD
    #        - the ref: Giseop Kim, Byungjae Park and Ayoung Kim, 1-Day Learning, 1-Year Localization: Long-term LiDAR Localization using Scan Context Image. IEEE Robotics and Automation Letters (RA-L) (with ICRA), 4(2):1948-1955, 2019. 
    roi_scale = 25
    roi = np.array([[-roi_scale, roi_scale], [-roi_scale, roi_scale], [-roi_scale, roi_scale]])
    scan = parse_roi_points(np.asarray(pcd.points), roi)
    
    # removing ground plane points 
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scan)
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.2,
                                         ransac_n=3,
                                         num_iterations=1000)
    noneplane_cloud = pcd.select_by_index(inliers, invert=True)
    viz = 0
    if viz:
        plane_cloud = pcd.select_by_index(inliers)
        plane_cloud.paint_uniform_color([1.0, 0, 0])
        noneplane_cloud.paint_uniform_color([0, 0, 1.0])
        o3d.visualization.draw_geometries([plane_cloud, noneplane_cloud])

    # sampling 4096 points 
    scan = np.asarray(noneplane_cloud.points)
    sampled_index = np.random.choice(scan.shape[0], num_pnvlad_points, replace=False)      
    scan = scan[sampled_index, :]

    # rescaling into [-1, 1] with zero mean 
    scan_pnvlad = rescaling_points(scan, roi_scale)
        
    # pcd to npy 
    print("processed point cloud:", scan_pnvlad.shape )
    
    # save
    save_path = os.path.join(output_dir, 'npy', reformat_extension(scan_name, "npy"))
    np.save(save_path, scan_pnvlad) # will be fed to the pointnetvlad's evaluation.py 

    savepcd = o3d.geometry.PointCloud()
    savepcd.points = o3d.utility.Vector3dVector(scan_pnvlad)
    save_path = os.path.join(output_dir, 'ply', reformat_extension(scan_name, "ply"))
    o3d.io.write_point_cloud(save_path, savepcd) # ply for fast debugging in CloudCompare 

### Take three COLMAP models, perform point cloud registration using Open3D ICP,
### and transform model0 to align with model1, transform model2 to align with model1.
### Visualize all three aligned models together colored by sequence.


import os
import sys
import copy
import numpy as np
import open3d as o3d

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLMAP_PYTHON_DIR = PROJECT_ROOT / "baselines" / "VSLAM_LAB" / "Baselines" / "colmap" / "scripts" / "python"

for path in (PROJECT_ROOT, COLMAP_PYTHON_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from read_write_model import read_model, qvec2rotmat, rotmat2qvec
from constants import plotting_parameters

BLUE = [106, 178, 212]
YELLOW = [243, 201, 9]
PINK = [227, 119, 194]
GREY = [127, 127, 127]

blue   = [c / 255.0 for c in BLUE]
yellow = [c / 255.0 for c in YELLOW]
pink   = [c / 255.0 for c in PINK]
grey   = [c / 255.0 for c in GREY]

def colmap_to_o3d(points3D):
    xyz = np.array([p.xyz for p in points3D.values()])
    rgb = np.array([p.rgb for p in points3D.values()]) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color(yellow) # [1, 0.706, 0]
    target_temp.paint_uniform_color(blue) # [0, 0.651, 0.929]
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(source, target, voxel_size):
    print(":: Load two point clouds.")

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 5 # voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(10000000, 0.999))
    return result

def get_transformation(pcd0, pcd1):  # source, target
    source = pcd0 #o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
    target = pcd1 #o3d.io.read_point_cloud(demo_icp_pcds.paths[1])
    
    # o3d.visualization.draw_geometries([source, target])

    voxel_size = 0.05  # 0.05 means 5cm for this dataset
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
        source, target, voxel_size)
    
    # perform global registration with identity matrix as initial transform
    result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)
    print(result_ransac)

    print(f"\nT = {result_ransac.transformation}")
    # draw_registration_result(source, target, result_ransac.transformation)
    
    # from the transformation matrix, extract R, t, s
    T = result_ransac.transformation
    return T

def transform_source(images0, points3D0, T):
    A = T[:3, :3]
    
    R = A
    t = T[:3, 3]
    s = 1.0
    
    print(f"R = {R}")
    print(f"t = {t}")
    print(f"s = {s}")
        
    # transform 3D points
    points3D_aligned = copy.deepcopy(points3D0)
    for pt in points3D_aligned.values():                
        xyz = s * (R @ pt.xyz) + t
        pt.xyz[0], pt.xyz[1], pt.xyz[2] = xyz[0], xyz[1], xyz[2]
    
    # transform camera poses
    images0_aligned = copy.deepcopy(images0)
    for img in images0_aligned.values():
        R_cw = qvec2rotmat(img.qvec)
        t_cw = img.tvec

        # convert world-to-camera → camera-to-world
        R_wc = R_cw.T
        t_wc = -R_wc @ t_cw

        # apply Umeyama transform
        R_wc_rotated = R @ R_wc
        t_wc_rotated = s * (R @ t_wc) + t

        # convert back to world-to-camera
        R_cw_rotated = R_wc_rotated.T
        t_cw_rotated = -R_cw_rotated @ t_wc_rotated

        img.qvec[:] = rotmat2qvec(R_cw_rotated)
        img.tvec[0], img.tvec[1], img.tvec[2] = t_cw_rotated[0], t_cw_rotated[1], t_cw_rotated[2]
    return images0_aligned, points3D_aligned

if __name__ == "__main__":

    model0_path = Path(f"/media/beverley/beverley_t7/VSLAM-LAB-Evaluation/exp_eff_full_icp/EIFFEL/eff16-full/colmap_00000/0")
    model1_path = Path(f"/media/beverley/beverley_t7/VSLAM-LAB-Evaluation/exp_eff_full_icp/EIFFEL/eff18-full/colmap_00000/0")
    model2_path = Path(f"/media/beverley/beverley_t7/VSLAM-LAB-Evaluation/exp_eff_full_icp/EIFFEL/eff20-full/colmap_00000/0")

    for path in [model0_path, model1_path, model2_path]:
            if not path.exists():
                print(f"⚠️  Path does not exist: {path}")
                exit(0)
    
    yellow_text = "\033[93m"
    reset_text = "\033[0m"
    print(f"{yellow_text}MODEL 0: {model0_path}{reset_text}")
    print(f"{yellow_text}MODEL 1: {model1_path}{reset_text}")
    print(f"{yellow_text}MODEL 2: {model2_path}{reset_text}")
    confirm = input(f"Please confirm that the above hard-coded paths are correct (Y/n): ").strip().lower()
    if confirm not in ["", "y"]:
        print("Exiting. Please edit the paths in the script and re-run.")
        exit(0)

    # load models
    cameras0, images0, points3D0 = read_model(model0_path, ext='.bin')
    cameras1, images1, points3D1 = read_model(model1_path, ext='.bin')
    cameras2, images2, points3D2 = read_model(model2_path, ext='.bin')

    # convert colmap points3D to open3d pointclouds
    pcd0 = colmap_to_o3d(points3D0)
    pcd1 = colmap_to_o3d(points3D1)
    pcd2 = colmap_to_o3d(points3D2)
    
    T0 = get_transformation(pcd0, pcd1)
    np.save(model0_path.parent.parent / "T.npy", T0)

    np.save(model1_path.parent.parent / "T.npy", np.eye(4))
    
    T2 = get_transformation(pcd2, pcd1) # 2018 source to 2017 target
    np.save(model2_path.parent.parent / "T.npy", T2)
    
        

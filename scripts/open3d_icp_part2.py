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

from read_write_model import read_model, qvec2rotmat, rotmat2qvec, write_model
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

    for path in [model0_path, model1_path, model2_path]:
        cameras, images, points3D = read_model(path, ext='.bin')
        pcd = colmap_to_o3d(points3D)
        
        T = np.load(path.parent.parent / "T.npy")
        print(f"\nLoaded T from {path.parent.parent / 'T.npy'}:\n{T}\n")
    
        images_aligned, points3D_aligned = transform_source(images, points3D, T)

        # save aligned model
        model_root = path.parent
        aligned_output_path = model_root / "0_transformed"
        os.makedirs(aligned_output_path, exist_ok=True)
        print(f"\n[💾] Writing aligned COLMAP model to: {aligned_output_path}\n")
        write_model(cameras, images_aligned, points3D_aligned, aligned_output_path, ext=".bin")

    # sanity check: visualize all three aligned models together
    cameras0_aligned, images0_aligned, points3D_aligned = read_model(Path(f'{model0_path}_transformed'), ext='.bin')
    cameras1_aligned, images1_aligned, points3D1_aligned = read_model(Path(f'{model1_path}_transformed'), ext='.bin')
    cameras2_aligned, images2_aligned, points3D2_aligned = read_model(Path(f'{model2_path}_transformed'), ext='.bin')
    
    points2016 = colmap_to_o3d(points3D_aligned)
    points2017 = colmap_to_o3d(points3D1_aligned)
    points2018 = colmap_to_o3d(points3D2_aligned)
    
    points2016.paint_uniform_color(blue)
    points2017.paint_uniform_color(yellow)
    points2018.paint_uniform_color(pink)

    o3d.visualization.draw_geometries([points2016, points2017, points2018])
    

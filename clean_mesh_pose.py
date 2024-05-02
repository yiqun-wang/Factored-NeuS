import numpy as np
import cv2 as cv
import os
from glob import glob
from scipy.io import loadmat
import trimesh

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--scene', type=str, required=True)  # 65
parser.add_argument('--setting', type=str, required=True)  # womask/geometry
parser.add_argument("--suffix", default="")  # 00300000
# parser.add_argument("--confname")

args = parser.parse_args()
scans = [int(args.scene)]
base_name = args.setting
# expname = Path(args.confname).with_suffix("").name

suffix = int(args.suffix)

def clean_points_by_mask(points, scan):
    cameras = np.load('./public_data/data_DTU/dtu_scan{}/cameras_sphere.npz'.format(scan))
    mask_lis = sorted(glob('./public_data/data_DTU/dtu_scan{}/mask/*.png'.format(scan)))
    n_images = 49 if scan < 83 else 64
    inside_mask = np.ones(len(points)) > 0.5
    for i in range(n_images):
        P = cameras['world_mat_{}'.format(i)]
        pts_image = np.matmul(P[None, :3, :3], points[:, :, None]).squeeze() + P[None, :3, 3]
        pts_image = pts_image / pts_image[:, 2:]
        pts_image = np.round(pts_image).astype(np.int32) + 1

        mask_image = cv.imread(mask_lis[i])
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (25, 25))
        mask_image = cv.dilate(mask_image, kernel, iterations=1)
        mask_image = (mask_image[:, :, 0] > 128)

        mask_image = np.concatenate([np.ones([1, 1600]), mask_image, np.ones([1, 1600])], axis=0)
        mask_image = np.concatenate([np.ones([1202, 1]), mask_image, np.ones([1202, 1])], axis=1)

        curr_mask = mask_image[(pts_image[:, 1].clip(0, 1201), pts_image[:, 0].clip(0, 1601))]

        inside_mask &= curr_mask.astype(bool)

    return inside_mask


for scan in scans:
    old_dir = f"./exp/data_DTU/dtu_scan{str(scan)}/{base_name}/meshes/"
    new_dir = f"./exp/data_DTU/dtu_scan{str(scan)}/{base_name}/meshes_clean/"
    os.makedirs(new_dir, exist_ok=True)
    print(scan)
    old_mesh = trimesh.load(os.path.join(old_dir, '{:0>8d}.ply'.format(suffix)))
    old_vertices = old_mesh.vertices[:]
    old_faces = old_mesh.faces[:]
    mask = clean_points_by_mask(old_vertices, scan)
    indexes = np.ones(len(old_vertices)) * -1
    indexes = indexes.astype(np.long)
    indexes[np.where(mask)] = np.arange(len(np.where(mask)[0]))

    faces_mask = mask[old_faces[:, 0]] & mask[old_faces[:, 1]] & mask[old_faces[:, 2]]
    new_faces = old_faces[np.where(faces_mask)]
    new_faces[:, 0] = indexes[new_faces[:, 0]]
    new_faces[:, 1] = indexes[new_faces[:, 1]]
    new_faces[:, 2] = indexes[new_faces[:, 2]]
    new_vertices = old_vertices[np.where(mask)]

    new_mesh = trimesh.Trimesh(new_vertices, new_faces)
    
    meshes = new_mesh.split(only_watertight=False)
    new_mesh = meshes[np.argmax([len(mesh.faces) for mesh in meshes])]

    new_mesh.export(os.path.join(new_dir, '{:0>8d}.ply'.format(suffix)))

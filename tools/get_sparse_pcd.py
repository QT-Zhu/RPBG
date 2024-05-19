import numpy as np
import trimesh
import pycolmap
import click

@click.command()
@click.option('--colmap_recons', type=str)
@click.option('--output_file', type=str)
def extract_sparse_pcd(colmap_recons, output_file):
    recons = pycolmap.Reconstruction(colmap_recons)
    points3D = recons.points3D

    xyz = []
    color = []
    for pid in points3D:
        point = points3D[pid]
        xyz.append(point.xyz)
        color.append(point.color)

    xyz = np.array(xyz)
    color = np.column_stack([np.array(color), 255 * np.ones(xyz.shape[0])])

    pcd = trimesh.PointCloud(vertices=xyz, colors=color)
    pcd.export(output_file)

if __name__ == "__main__":
    extract_sparse_pcd()

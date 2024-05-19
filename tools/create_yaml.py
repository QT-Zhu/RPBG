from pathlib import Path
import click
from pose_format_converter import *

@click.command()
@click.option('--datapath', type=str)
@click.option('--colmap_sparse', type=str)
def create_yaml(datapath, colmap_sparse):
    traj = load_colmap(colmap_sparse)
    traj.export_agisoft(str(Path(datapath)/"camera.xml"))
    W, H = traj.width, traj.height
    sparse_yaml = [
        f"viewport_size: [{W}, {H}]\n",
        "intrinsic_matrix: camera.xml\n",
        "view_matrix: camera.xml\n",
        "pointcloud: sfm/sparse_pcd.ply"
    ]
    dense_yaml = [
        f"viewport_size: [{W}, {H}]\n",
        "intrinsic_matrix: camera.xml\n",
        "view_matrix: camera.xml\n",
        "pointcloud: mvs/dense_pcd.ply"
    ]
    with open(str(Path(datapath)/"scene-sparse.yaml"), "w") as f:
        f.writelines(sparse_yaml)
    with open(str(Path(datapath)/"scene-dense.yaml"), "w") as f:
        f.writelines(dense_yaml)


if __name__ == "__main__":
    create_yaml()
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
import open3d
import trimesh
import pycolmap
from pathlib import Path
import cv2


def intrinsics_extrinsics_from_txt(txt_file):
    with open(txt_file) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    return intrinsics, extrinsics

def intrinsics_from_xml(xml_file):
    root = ET.parse(xml_file).getroot()
    calibration = root.find('chunk/sensors/sensor/calibration')
    resolution = calibration.find('resolution')
    width = float(resolution.get('width'))
    height = float(resolution.get('height'))
    try:
        f = float(calibration.find('f').text)
        fx = fy = f
    except:
        fx = float(calibration.find('fx').text)
        fy = float(calibration.find('fy').text)

    try:
        cx = float(calibration.find('cx').text)
        cy = float(calibration.find('cy').text)
    except:
        cx, cy = width/2, height/2

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0,  1]
        ], dtype=np.float32)

    return K, (width, height)

# opencv/open3d convention
def extrinsics_from_xml(xml_file):
    root = ET.parse(xml_file).getroot()
    transforms = {}
    for e in root.findall('chunk/cameras')[0].findall('camera'):
        label = e.get('label')
        transforms[label] = e.find('transform').text
    view_matrices = []

    labels_sort = list(transforms)
    for label in labels_sort:
        extrinsic = np.array([float(x) for x in transforms[label].split()]).reshape(4, 4)
        extrinsic = np.linalg.inv(extrinsic)
        view_matrices.append(extrinsic)

    return view_matrices, labels_sort

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

class MyTrajectory:
    def __init__(self, K, width, height, extrinsics, labels=None):
        self.K = K
        self.width = int(width)
        self.height = int(height)
        self.extrinsics = extrinsics # in opencv/open3d convention
        if labels:
            assert len(extrinsics) == len(labels)
            self.labels = labels
        else:
            self.labels = [f"{i:08d}" for i in range(len(extrinsics))]
    
    def export_open3d(self, filename):
        open3d_cams = []
        for T in self.extrinsics:
            cam = open3d.camera.PinholeCameraParameters()
            cam.extrinsic = T
            cam_intrinsic = open3d.camera.PinholeCameraIntrinsic(width=int(self.width),height=int(self.height),
                                                                 fx=self.K[0,0],fy=self.K[1,1],cx=self.K[0,2],cy=self.K[1,2])
            cam.intrinsic = cam_intrinsic 
            open3d_cams.append(cam)
        traj = open3d.camera.PinholeCameraTrajectory()
        traj.parameters = open3d_cams
        open3d.io.write_pinhole_camera_trajectory(str(filename), traj)

    def export_agisoft(self, filename):
        document = ET.Element('document')
        chunk = ET.SubElement(document, "chunk")
        cameras = ET.SubElement(chunk, "cameras")
        sensors = ET.SubElement(chunk, 'sensors')
        sensor = ET.SubElement(sensors, 'sensor')
        calibration = ET.SubElement(sensor, 'calibration')
        resolution = ET.SubElement(calibration, 'resolution', \
                                width=str(self.width), \
                                height=str(self.height)
                                )
        fx = ET.SubElement(calibration,"fx")
        fx.text = str(self.K[0,0])
        fy = ET.SubElement(calibration,"fy")
        fy.text = str(self.K[1,1])
        cx = ET.SubElement(calibration, "cx")
        cx.text = str(self.K[0,2])
        cy = ET.SubElement(calibration, "cy")
        cy.text = str(self.K[1,2])

        for T, label in zip(self.extrinsics, self.labels):
            cam_tag = ET.SubElement(cameras, "camera", label=label)
            transform = ET.SubElement(cam_tag, "transform")
            transform_text = " ".join([str(x) for x in np.linalg.inv(T).reshape(16)])
            transform.text = transform_text

        xmlstr = minidom.parseString(ET.tostring(document)).toprettyxml(indent="   ")
        with open(str(filename), "w") as f:
            f.write(xmlstr)

    def export_colmap(self, sparse_path, img_postfix=".jpg"):
        sparse_path = Path(sparse_path)
        sparse_path.mkdir(exist_ok=True, parents=True)

        # write cameras.txt
        camera_lines = [f"1 PINHOLE {self.width} {self.height} {self.K[0,0]} {self.K[1,1]} {self.K[0,2]} {self.K[1,2]}"]
        with open(str(sparse_path/"cameras.txt"), "w") as f:
            f.writelines(camera_lines)

        # write images.txt
        # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        image_lines = []
        for i, (T, label) in enumerate(zip(self.extrinsics, self.labels)):
            qvec = rotmat2qvec(T[:3,:3])
            tvec = T[:3,-1]
            image_line = f"{i+1} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {tvec[0]} {tvec[1]} {tvec[2]} 1 {label+img_postfix}\n"
            image_lines.append(image_line)
            image_lines.append("\n")
        with open(str(sparse_path/"images.txt"), "w") as f:
            f.writelines(image_lines)

        # write points3D.txt (empty)
        with open(str(sparse_path/"points3D.txt"), "w") as f:
            f.writelines([])

    def export_pointcloud(self, filename):
        xyz = [-T[:3,:3].T @ T[:3,-1] for T in self.extrinsics]
        pcd = trimesh.PointCloud(np.array(xyz))     
        pcd.export(filename)   

def load_agisoft(agisoft_file, eval_list=None):
    K, (width, height) = intrinsics_from_xml(str(agisoft_file))
    extrinsics, labels = extrinsics_from_xml(str(agisoft_file))
    if not eval_list:
        traj = MyTrajectory(K, width, height, extrinsics, labels)
    else:
        with open(eval_list, "r") as f:
            eval_labels = [Path(line).stem for line in f.readlines()]
        eval_traj = {}
        for T, label in zip(extrinsics, labels):
            if label in eval_labels:
                eval_traj[label] = T
        assert len(eval_labels) == len(eval_traj)
        
        traj = MyTrajectory(K, width, height, [eval_traj[label] for label in sorted(eval_traj)], [label for label in sorted(eval_traj)])

    return traj

# use first intrinsic
def load_open3d(open3d_file):
    cams_list = open3d.io.read_pinhole_camera_trajectory(str(open3d_file)).parameters
    K, width, height = cams_list[0].intrinsic.intrinsic_matrix, cams_list[0].intrinsic.width, cams_list[0].intrinsics.height
    extrinsics = [cam.extrinsic for cam in cams_list]
    traj = MyTrajectory(K, width, height, extrinsics)
    return traj

# use first intrinsic (camera_id=1)
def load_colmap(sparse_path):
    recons = pycolmap.Reconstruction(sparse_path)
    cameras, images = recons.cameras, recons.images
    K = cameras[1].calibration_matrix()
    height, width = cameras[1].height, cameras[1].width
    extrinsics = []
    labels = []
    for image_id in images:
        image = images[image_id]
        M = image.projection_matrix()
        T = np.row_stack([M,[0,0,0,1]])
        file_postfix = image.name.split(".")[-1]
        label = image.name.replace("."+file_postfix,"")
        print(label)
        labels.append(label) # remove postfix
        extrinsics.append(T)

    traj = MyTrajectory(K, width, height, extrinsics, labels)
    return traj

# use first intrinsic
def load_mvs(mvs_path):
    mvs_path = Path(mvs_path)
    image_path = mvs_path / "images"
    cam_path = mvs_path / "cams"
    sample_img = cv2.imread(str(list(image_path.iterdir())[0]))
    H, W, _ = sample_img.shape

    extrinsics = []
    labels = []
    
    for cam in sorted(cam_path.iterdir()):
        labels.append(cam.stem.replace("_cam",""))
        K, T = intrinsics_extrinsics_from_txt(str(cam))
        extrinsics.append(T)
    traj = MyTrajectory(K, W, H, extrinsics, labels)
    return traj



if __name__ == "__main__":
    
    traj = load_agisoft(f"/data/data_RPBG/Mill19/building-pixsfm/camera.xml")
    traj.export_open3d(f"./building.json")


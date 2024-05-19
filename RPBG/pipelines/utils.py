import configparser
import os
import cv2
import numpy as np
import yaml
import trimesh
import xml.etree.ElementTree as ET


def recalc_proj_matrix_planes(pm, new_near=.01, new_far=1000.):
    depth = float(new_far - new_near)
    q = -(new_far + new_near) / depth
    qn = -2 * (new_far * new_near) / depth

    out = pm.copy()

    # Override near and far planes
    out[2, 2] = q
    out[2, 3] = qn

    return out

def get_proj_matrix(K, image_size, znear=.01, zfar=1000.):
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]
    width, height = image_size
    m = np.zeros((4, 4))
    m[0][0] = 2.0 * fx / width
    m[0][1] = 0.0
    m[0][2] = 0.0
    m[0][3] = 0.0

    m[1][0] = 0.0
    m[1][1] = 2.0 * fy / height
    m[1][2] = 0.0
    m[1][3] = 0.0

    m[2][0] = 1.0 - 2.0 * cx / width
    m[2][1] = 2.0 * cy / height - 1.0
    m[2][2] = (zfar + znear) / (znear - zfar)
    m[2][3] = -1.0

    m[3][0] = 0.0
    m[3][1] = 0.0
    m[3][2] = 2.0 * zfar * znear / (znear - zfar)
    m[3][3] = 0.0

    return m.T

def intrinsics_from_xml(xml_file):
    root = ET.parse(xml_file).getroot()
    calibration = root.find('chunk/sensors/sensor/calibration')
    resolution = calibration.find('resolution')
    width = float(resolution.get('width'))
    height = float(resolution.get('height'))
    try:
        # metashape output format
        f = float(calibration.find('f').text)
        fx = fy = f
        cx, cy = width/2, height/2
    except:
        # pseudo manual format
        fx = float(calibration.find('fx').text)
        fy = float(calibration.find('fy').text)
        cx = float(calibration.find('cx').text)
        cy = float(calibration.find('cy').text)
    # f = float(calibration.find('f').text)
    # cx = width/2
    # cy = height/2
        
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0,  1]
        ], dtype=np.float32)

    return K, (width, height)

def intrinsics_from_ini(ini_path):
    conf = configparser.ConfigParser()
    conf.read(ini_path)
    K_info = np.fromstring(conf.get('SceneCameraParams', 'K'), sep=' ')
    K = np.array([
        [K_info[0], 0, K_info[2]],
        [0,K_info[1], K_info[3]],
        [0,0,1]
    ],dtype = np.float32)
    img_size = [int(conf.get('SceneCameraParams', 'w')), int(conf.get('SceneCameraParams', 'h'))]
    print(K)
    return K, img_size

def intrinsics_from_txt(cam_txt):
    tmp = np.loadtxt(cam_txt)
    img_size = [int(tmp[0,0]),int(tmp[0,1])]
    K = tmp[1:,].astype(np.float32)
    return K, img_size

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
        extrinsic[:, 1:3] *= -1
        view_matrices.append(extrinsic)

    return view_matrices, labels_sort

def extrinsics_from_view_matrix(path):
    vm = np.loadtxt(path).reshape(-1,4,4)
    vm, ids = get_valid_matrices(vm)

    # we want consistent camera label data type, as cameras from xml
    ids = [str(i) for i in ids]

    return vm, ids

def extrinsics_from_txt(pose_path):
    extrins = np.loadtxt(pose_path).reshape(-1,4,4)
    extrins[:,:,1:3] *=-1
    img_path = pose_path.split('/')
    img_path[-1] = img_path[-1].replace('poses', 'images')
    img_path = '/'.join(img_path)
    img_names = np.loadtxt(img_path, dtype='str')
    return extrins, img_names

def load_scene_data(path):
    with open(path, 'r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    if 'pointcloud' in config:
        print('loading pointcloud...')
        pointcloud = import_model3d(fix_relative_path(config['pointcloud'], path))
    else:
        pointcloud = None

    if 'mesh' in config and config['mesh']:
        print('loading mesh...')
        uv_order = config['uv_order'] if 'uv_order' in config else 's,t'
        mesh = import_model3d(fix_relative_path(config['mesh'], path), uv_order=uv_order.split(','), is_mesh=True)
    else:
        mesh = None

    if config.get('texture'):
        print('loading texture...')
        texture = cv2.imread(fix_relative_path(config['texture'], path))
        assert texture is not None
        texture = texture[..., ::-1].copy()
    else:
        texture = None

    if 'intrinsic_matrix' in config:
        apath = fix_relative_path(config['intrinsic_matrix'], path)
        if apath[-3:] == 'xml':
            intrinsic_matrix, (width, height) = intrinsics_from_xml(apath)
            #assert tuple(config['viewport_size']) == (width, height), f'calibration width, height: ({width}, {height})'
        elif apath[-3:] == 'ini':
            intrinsic_matrix, (width, height) = intrinsics_from_ini(apath)
        elif apath[-3:] == 'txt':
            intrinsic_matrix, (width, height) = intrinsics_from_txt(apath)
        else:
            intrinsic_matrix = np.loadtxt(apath)[:3, :3]
    else:
        intrinsic_matrix = None
    
    if 'proj_matrix' in config:
        proj_matrix = np.loadtxt(fix_relative_path(config['proj_matrix'], path))
        proj_matrix = recalc_proj_matrix_planes(proj_matrix)
    else:
        proj_matrix = None

    if 'view_matrix' in config:
        apath = fix_relative_path(config['view_matrix'], path)
        if apath[-3:] == 'xml':
            view_matrix, camera_labels = extrinsics_from_xml(apath)
        elif apath[-3:] == 'txt':
            view_matrix, camera_labels = extrinsics_from_txt(apath)
        else:
            view_matrix, camera_labels = extrinsics_from_view_matrix(apath)
    else:
        view_matrix = None
    # print(camera_labels)

    if 'model3d_origin' in config:
        model3d_origin = np.loadtxt(fix_relative_path(config['model3d_origin'], path))
    else:
        model3d_origin = np.eye(4)

    if 'point_sizes' in config:
        point_sizes = np.load(fix_relative_path(config['point_sizes'], path))
    else:
        point_sizes = None
    config['viewport_size'] = tuple(config['viewport_size'])


    if 'net_path' in config:
        net_ckpt = os.path.join(config['net_path'], 'checkpoints', config['ckpt'])
        net_ckpt = fix_relative_path(net_ckpt, path)

        tex_ckpt = os.path.join(config['net_path'], 'checkpoints', config['texture_ckpt'])
        tex_ckpt = fix_relative_path(tex_ckpt, path)
    else:
        net_ckpt = ""
        tex_ckpt = ""

    if 'data_ratio' in config:
        data_ratio = config['data_ratio']
        l = int(len(view_matrix) * data_ratio)
        view_matrix = view_matrix[:l]
        camera_labels = camera_labels[:l]

    return {
        'pointcloud': pointcloud,
        'point_sizes': point_sizes,
        'mesh': mesh,
        'texture': texture,
        'proj_matrix': proj_matrix,
        'intrinsic_matrix': intrinsic_matrix,
        'view_matrix': view_matrix,
        'camera_labels': camera_labels,     
        'model3d_origin': model3d_origin,
        'config': config,
        'net_ckpt': net_ckpt,
        'tex_ckpt': tex_ckpt
    }

def fix_relative_path(path, config_path):
    if not os.path.exists(path) and not os.path.isabs(path):
        root = os.path.dirname(config_path)
        abspath = os.path.join(root, path)
        if os.path.exists(abspath):
            return abspath
    return path

def get_valid_matrices(mlist):
    ilist = []
    vmlist = []
    for i, m in enumerate(mlist):
        if np.isfinite(m).all():
            ilist.append(i)
            vmlist.append(m)

    return vmlist, ilist

def get_xyz_colors(xyz, r=8):
    mmin, mmax = xyz.min(axis=0), xyz.max(axis=0)
    color = (xyz - mmin) / (mmax - mmin)
    # color = 0.5 + 0.5 * xyz / r
    return np.clip(color, 0., 1.).astype(np.float32)

def import_model3d(model_path, uv_order=None, is_mesh=False):
    data = trimesh.load(model_path)

    n_pts = data.vertices.shape[0]

    model = {
        'rgb': None,
        'normals': None,
        'uv2d': None,
        'faces': None
    }

    if is_mesh:
        if hasattr(data.visual, 'vertex_colors'):
            model['rgb'] = data.visual.vertex_colors[:, :3] / 255.
        elif hasattr(data.visual, 'to_color'):
            try:
                # for some reason, it may fail (happens on blender exports)
                model['rgb'] = data.visual.to_color().vertex_colors[:, :3] / 255.
            except:
                print('data.visual.to_color failed')

        model['normals'] = data.vertex_normals

        if hasattr(data.visual, 'uv'):
            model['uv2d'] = data.visual.uv
        # elif model_path[-3:] == 'ply':
        #     mdata = data.metadata['ply_raw']['vertex']['data']
        #     if 's' in mdata and 't' in mdata:
        #         print('using s,t texture coords')
        #         model['uv2d'] = np.hstack([mdata['s'], mdata['t']])
        #         print(model['uv2d'].shape)

        model['faces'] = data.faces.flatten().astype(np.uint32)
    else:
        if hasattr(data, 'colors'):
            if len(data.colors) >0 :
                model['rgb'] = data.colors[:, :3] / 255.  
        else:
            try:
                model['rgb'] = data.visual.vertex_colors[:, :3] / 255.
            except:
                pass
            
        if 'ply_raw' in data.metadata:
            normals = np.zeros((n_pts, 3), dtype=np.float32)
            if hasattr(data.metadata['ply_raw']['vertex']['data'], 'nx'):
                normals[:, 0] = data.metadata['ply_raw']['vertex']['data']['nx']
                normals[:, 1] = data.metadata['ply_raw']['vertex']['data']['ny']
                normals[:, 2] = data.metadata['ply_raw']['vertex']['data']['nz']
                model['normals'] = normals
        elif hasattr(data, 'vertex_normals'):
            model['normals'] = data.vertex_normals

        model['uv2d'] = np.zeros((n_pts, 2), dtype=np.float32)

    model['xyz'] = data.vertices

    model['xyz_c'] = get_xyz_colors(data.vertices)
    model['uv1d'] = np.arange(n_pts)

    if model['rgb'] is None:
        print(f'no colors in {model_path}')
        model['rgb'] = np.ones((n_pts, 3), dtype=np.float32)*255 

    if model['uv2d'] is None:
        if is_mesh:
            print(f'no uv in {model_path}')
        model['uv2d'] = np.zeros((n_pts, 2), dtype=np.float32)

    if model['faces'] is None:
        if is_mesh:
            print(f'no faces in {model_path}')
        model['faces'] = np.array([0, 1, 2], dtype=np.uint32)

    print('=== 3D model ===')
    print('VERTICES: ', n_pts)
    print('EXTENT: ', model['xyz'].min(0), model['xyz'].max(0))
    print('================')

    return model


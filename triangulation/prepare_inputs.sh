set -e

root_path=<REPLACE_TO_YOUR_ROOT>
colmap_exec=<YOUR_COLMAP_PATH>

local_path=$(pwd)

mkdir -p $root_path/sfm

cd $root_path

echo ---------1/3: Start Sparse Triangulation----------

$colmap_exec feature_extractor \
    --database_path sfm/database.db \
    --image_path images \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model SIMPLE_RADIAL \
    --SiftExtraction.gpu_index 0

$colmap_exec exhaustive_matcher \
    --database_path sfm/database.db \
    --SiftMatching.gpu_index 0

$colmap_exec mapper \
    --database_path sfm/database.db \
    --image_path images \
    --output_path sfm

$colmap_exec image_undistorter \
    --image_path images \
    --input_path sfm/0 \
    --output_path sfm/undis

$colmap_exec model_converter \
    --input_path sfm/undis/sparse \
    --output_path sfm/undis/sparse \
    --output_type TXT

cd $local_path

python tools/get_sparse_pcd.py --colmap_recons $root_path/sfm/0 --output_file $root_path/sfm/sparse_pcd.ply


echo ---------2/3: Start Dense Triangulation----------

dense_folder=$root_path/mvs

python triangulation/mvs/colmap2mvsnet.py \
       --dense_folder $dense_folder \
       --image_folder $root_path/sfm/undis/images \
       --sfm_folder $root_path/sfm/undis/sparse

view_num=5

output_folder=$dense_folder/outputs

CUDA_VISIBLE_DEVICES=0 python triangulation/mvs/eval.py \
       --dataset=data_eval_custom \
       --batch_size=4 \
       --inverse_depth=True \
       --numdepth=256 \
       --view_num=$view_num \
       --max_h=600 \
       --max_w=800 \
       --image_scale=1.0 \
       --testpath=$dense_folder \
       --loadckpt=triangulation/mvs/model_blended_v2.ckpt \
       --outdir=$output_folder

python triangulation/mvs/fusion.py \
       --photo_threshold 0.5 \
       --testpath=$dense_folder \
       --outdir=$output_folder \
       --view_num $view_num

echo ---------3/3: Start Configuration Generation----------

python tools/create_yaml.py --datapath $root_path --colmap_sparse $root_path/sfm/undis/sparse

echo ---------Finish! Please follow the instructions in README!----------
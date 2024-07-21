#!/bin/bash
set -xe

#PRJ_DIR="/home/sangjun/soon/FOUND/data/scans/0048_c"
PRJ_DIR="/home/sangjun/soon/FOUND/data/scans/test4"
IMAGES_PATH="${PRJ_DIR}/rgb"
DATASET_PATH="${PRJ_DIR}/colmap"
rm -rf ${DATASET_PATH}
mkdir -p ${DATASET_PATH}

#colmap automatic_reconstructor \
#    --workspace_path ${DATASET_PATH} \
#    --image_path ${IMAGES_PATH}

echo "images_path: ${IMAGES_PATH}"
echo "dataset_path: ${DATASET_PATH}"

colmap feature_extractor \
 --database_path $DATASET_PATH/database.db \
 --image_path $IMAGES_PATH \
 --ImageReader.single_camera 1

colmap exhaustive_matcher \
 --database_path $DATASET_PATH/database.db

rm -rf $DATASET_PATH/sparse
mkdir -p $DATASET_PATH/sparse

colmap mapper \
  --database_path $DATASET_PATH/database.db \
  --image_path $IMAGES_PATH \
  --output_path $DATASET_PATH/sparse \
  --Mapper.ba_refine_focal_length 1

rm -rf $DATASET_PATH/sparse_text
mkdir -p $DATASET_PATH/sparse_text/0
colmap model_converter \
  --input_path=${DATASET_PATH}/sparse/0 \
  --output_path=${DATASET_PATH}/sparse_text/0 \
  --output_type TXT

rm -rf $DATASET_PATH/dense
mkdir -p $DATASET_PATH/dense

colmap image_undistorter \
  --image_path $IMAGES_PATH \
  --input_path $DATASET_PATH/sparse/0 \
  --output_path $DATASET_PATH/dense \
  --output_type COLMAP
  #--max_image_size 2000

# mkdir -p $DATASET_PATH/dense/stereo
# colmap patch_match_stereo \
#   --workspace_path $DATASET_PATH/dense \
#   --workspace_format COLMAP \
#   --PatchMatchStereo.geom_consistency 1
# 
# colmap stereo_fusion \
#   --workspace_path $DATASET_PATH/dense \
#   --workspace_format COLMAP \
#   --input_type geometric \
#   --output_path $DATASET_PATH/dense/fused.ply
# 
# colmap poisson_mesher \
#   --input_path $DATASET_PATH/dense/fused.ply \
#   --output_path $DATASET_PATH/dense/meshed-poisson.ply \
#   --PoissonMeshing.trim 1

# colmap delaunay_mesher \
#   --input_path $DATASET_PATH/dense \
#   --output_path $DATASET_PATH/dense/meshed-delaunay.ply
# 

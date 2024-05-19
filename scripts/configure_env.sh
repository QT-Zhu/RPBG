set -e

pip install -r requirements.txt
pip install ./pcpr

# optional: for headless rendering by Open3D
# conflit with tensorboard
# if not intended to render pcd for comparison, skip Open3D
conda install cmake -y
conda install -c conda-forge mesalib libglu -y


# if you don't need headless rendering
# pip install open3d


# if you would like to apply the headless rendering of Open3d 
# known issue with 0.17
git clone --depth 1 --branch v0.15.1 https://github.com/isl-org/Open3D.git

export LIBRARY_PATH=$LIBRARY_PATH:$CONDA_PREFIX/lib 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib 
export C_INCLUDE_PATH=$C_INCLUDE_PATH:$CONDA_PREFIX/include/ 
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$CONDA_PREFIX/include/

cd Open3D && mkdir build && cd build
cmake .. -DENABLE_HEADLESS_RENDERING=ON \
         -DBUILD_GUI=OFF \
         -DBUILD_WEBRTC=OFF \
         -DUSE_SYSTEM_GLEW=OFF \
         -DUSE_SYSTEM_GLFW=OFF \
         -Wno-dev

make -j16
make install-pip-package
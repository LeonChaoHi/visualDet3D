# copy Stereo 3D example config
CONFIG_FILE = "myyolo3d"
cd config
cp Stereo3D_example $CONFIG_FILE.py

## Modify config path
nano $CONFIG_FILE.py
cd ..

## Compute image database and anchors mean/std
# You can run ./launcher/det_precompute.sh without arguments to see helper documents
./launcher/det_precompute.sh config/$CONFIG_FILE.py train
./launcher/det_precompute.sh config/$CONFIG_FILE.py test

## run this if disparity map is needed, can be computed with point cloud or openCV BlockMatching
# You can run ./launcher/disparity_precompute.sh without arguments to see helper documents
./disparity_precompute.sh config/$CONFIG_FILE.py $IsUsingPointCloud

## train the model with one GPU
# You can run ./launcher/train.sh without arguments to see helper documents
./launcher/train.sh  --config/$CONFIG_FILE.py 0 $experiment_name # validation goes along

## produce validation/test result # we only support single GPU testing
# You can run ./launcher/eval.sh without arguments to see helper documents
./launcher/eval.sh --config/$CONFIG_FILE.py 0 $CHECKPOINT_PATH validation/test
```
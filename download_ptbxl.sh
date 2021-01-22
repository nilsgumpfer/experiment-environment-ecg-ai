# Download PTBXL data into snapshot directory
mkdir ./data/ptbxl/snapshots/ -p
cd ./data/ptbxl/snapshots/
gsutil -m cp -r gs://ptb-xl-1.0.1.physionet.org .
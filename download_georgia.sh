# Download Georgia 12 lead data into snapshot directory
mkdir ./data/georgia/snapshots/ -p
cd ./data/georgia/snapshots/
gsutil -m cp -r gs://physionetchallenge2021-public-datasets/WFDB_Ga.tar.gz .
tar -xf WFDB_Ga.tar.gz
rm WFDB_Ga.tar.gz
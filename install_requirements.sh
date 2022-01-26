apt-get update
apt-get install -y apt-transport-https
apt-get install -y systemd
apt-get install -y nano
apt-get install -y git
apt-get install -y curl
apt-get install -y libcurl4-openssl-dev libssl-dev
apt-get install -y libsm6 libxext6 libxrender-dev
apt-get install -y python3-graphviz

# Install google cloud API tools, necessary to load data from https://physionet.org/content/ptb-xl/
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
apt-get install -y apt-transport-https ca-certificates gnupg
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
apt-get update
apt-get install -y google-cloud-sdk

# Init google cloud API
gcloud init

# Install python requirements
pip3 install -r requirements.txt

# Export python path
echo "export PYTHONPATH='..:.:../..'" >> ~/.bashrc
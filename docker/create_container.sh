docker stop $(whoami)_exp_env_ub1804py36tf22
docker rm $(whoami)_exp_env_ub1804py36tf22
docker rmi ub1804py36tf22:v1.0
docker build -t ub1804py36tf22:v1.0 .
#docker run -it -d --gpus all --name $(whoami)_exp_env_ub1804py36tf22 ub1804py36tf22:v1.0 bash
docker run -it -d --name $(whoami)_exp_env_ub1804py36tf22 ub1804py36tf22:v1.0 bash
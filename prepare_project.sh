bash configure_git.sh
bash install_gcloud.sh
pip install -r requirements.txt
apt install unzip

apt install -y htop nvtop

gcloud auth application-default login --no-browser
wandb login

dvc pull ./data/*.dvc
# dvc pull ./pipeline/dvc.yaml

unzip ./data/ml_interview_task_data.zip

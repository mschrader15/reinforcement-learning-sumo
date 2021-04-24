CONFIG_PATH="./settings/PPO_4_24.json"

rm ~/train.out

source ../venv/bin/activate
nohup python train.py --config_path $CONFIG_PATH > ~/train.out 2>&1


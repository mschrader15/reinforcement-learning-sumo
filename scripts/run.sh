CONFIG_PATH="./settings/ES_4_27_FC.json"

rm ~/train.out

source ../venv/bin/activate
nohup python train.py --config_path $CONFIG_PATH > ~/train.out 2>&1


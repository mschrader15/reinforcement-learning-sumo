CONFIG_PATH="./settings/ES_4_25.json"

rm ~/train.out

source ../venv/bin/activate
nohup python train.py --config_path $CONFIG_PATH > ~/train.out 2>&1


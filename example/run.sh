CONFIG_PATH="./setting-files/ES_4_25.json"

source ../venv/bin/activate
# nohup python ../train.py --config_path $CONFIG_PATH > ~/train.out 2>&1
python ../train.py --config_path $CONFIG_PATH
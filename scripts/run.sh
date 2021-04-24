CONFIG_PATH="./settings/PPO_4_24.json"

source ../venv/bin/activate
nohup python train.py --config_path $CONFIG_PATH > run.out 2>&1

# tensorboard --log_dir "~/ray_results"
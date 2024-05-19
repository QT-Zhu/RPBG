set -e

config_file=$1

dehead=${config_file#configs/*}
scene=${dehead%*.yaml}
echo scene: $scene

CUDA_VISIBLE_DEVICES=0 python run.py --eval \
    --config $config_file --name $scene

CUDA_VISIBLE_DEVICES=0 python eval.py --scan $scene --vis False


modelspath=$PWD"/trained-models/crashed_high_energy_srt_combo_210981234_211644345/"
flowpath=$modelspath"flow_epoch_10.pth"
modelpath=$modelspath"model_epoch_10.pth"
configpath="/home/axel/i3/LLP-at-IceCube/reconstruction/transformer-encoder/configs/model_high_energy_no_percentiles.yaml"

python check_saved_models.py --transformer $modelpath --flow $flowpath --config $configpath


topfolder="/home/axel/i3/i3-pq-conversion-files/high_energy_srt_cleaned_combo_210981234_211644345/"
modelpath="/home/axel/i3/LLP-at-IceCube/reconstruction/transformer-encoder/examples/train_cnf/trained-models/high_energy_srt_combo_210981234_211644345_gggggt/model_final.pth"
flowpath="/home/axel/i3/LLP-at-IceCube/reconstruction/transformer-encoder/examples/train_cnf/trained-models/high_energy_srt_combo_210981234_211644345_gggggt/flow_final.pth"
configpath="/home/axel/i3/LLP-at-IceCube/reconstruction/transformer-encoder/configs/model_high_energy_no_percentiles.yaml"
normpath="/home/axel/i3/LLP-at-IceCube/reconstruction/transformer-encoder/configs/normalization_args.yaml"
filenamestart="srt"

python check_train_labels.py \
 --topfolder "$topfolder" \
 --modelpath "$modelpath" \
 --flowpath "$flowpath" \
 --modelconfig "$configpath" \
 --filenamestart "$filenamestart" \
 --normpath "$normpath" \

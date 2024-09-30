cd ..

epoch=250

topfolder="/home/axel/i3/i3-pq-conversion-files/high_energy_srt_cleaned_combo_210981234_211644345_211828206/"
modelpath=$PWD"/trained-models/clipped2_high_energy_srt_cleaned_combo_210981234_211644345_211828206_large/model_epoch_$epoch.pth"
flowpath=$PWD"/trained-models/clipped2_high_energy_srt_cleaned_combo_210981234_211644345_211828206_large/flow_epoch_$epoch.pth"
configpath="/home/axel/i3/LLP-at-IceCube/reconstruction/transformer-encoder/configs/model_high_energy_no_percentiles_large.yaml"
normpath="/home/axel/i3/LLP-at-IceCube/reconstruction/transformer-encoder/configs/normalization_args.yaml"
filenamestart="srt"

nevents=4

python visualize_predictions.py \
 --topfolder "$topfolder" \
 --modelpath "$modelpath" \
 --flowpath "$flowpath" \
 --modelconfig "$configpath" \
 --filenamestart "$filenamestart" \
 --nevents "$nevents" \
 --normpath "$normpath" \
 --plot2x2 \
#  --unlabeled \
#  --shuffle \
#  --predicttrain \

cd run_scripts
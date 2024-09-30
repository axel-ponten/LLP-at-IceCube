cd ..

topfolder="/home/axel/i3/i3-pq-conversion-files/srt_cleaned_IC86.2020_corsika.020904.198000.i3.zst/"
modelpath=$PWD"/trained-models/clipped2_high_energy_srt_cleaned_combo_210981234_211644345_211828206_large/model_epoch_250.pth"
flowpath=$PWD"/trained-models/clipped2_high_energy_srt_cleaned_combo_210981234_211644345_211828206_large/flow_epoch_250.pth"
configpath="/home/axel/i3/LLP-at-IceCube/reconstruction/transformer-encoder/configs/model_high_energy_no_percentiles_large.yaml"
normpath="/home/axel/i3/LLP-at-IceCube/reconstruction/transformer-encoder/configs/normalization_args.yaml"
filenamestart="srt"

python test_corner_bkg.py \
 --topfolder "$topfolder" \
 --modelpath "$modelpath" \
 --flowpath "$flowpath" \
 --modelconfig "$configpath" \
 --filenamestart "$filenamestart" \
 --normpath "$normpath" \

cd run_scripts
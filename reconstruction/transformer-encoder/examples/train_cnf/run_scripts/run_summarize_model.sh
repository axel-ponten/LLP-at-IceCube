cd ..

epoch=195

topfolder="/home/axel/i3/i3-pq-conversion-files/srt_allE_combo_213638022_213712969/"
modelpath=$PWD"/trained-models/allE_srt_213638022_213712969_small/model_best.pth"
flowpath=$PWD"/trained-models/allE_srt_213638022_213712969_small/flow_best.pth"
configpath="/home/axel/i3/LLP-at-IceCube/reconstruction/transformer-encoder/configs/model_high_energy_no_percentiles.yaml"
normpath="/home/axel/i3/LLP-at-IceCube/reconstruction/transformer-encoder/configs/normalization_args.yaml"
filenamestart="srt"

python summarize_performance.py \
 --topfolder "$topfolder" \
 --modelpath "$modelpath" \
 --flowpath "$flowpath" \
 --modelconfig "$configpath" \
 --filenamestart "$filenamestart" \
 --normpath "$normpath" \

cd run_scripts
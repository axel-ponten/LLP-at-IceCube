cd ..

topfolder="/home/axel/i3/i3-pq-conversion-files/srt_allE_combo_213638022_213712969/"
modelpath="/home/axel/i3/LLP-at-IceCube/reconstruction/transformer-encoder/examples/train_cnf/trained-models/allE_srt_213638022_213712969_large/model_best.pth"
flowpath="/home/axel/i3/LLP-at-IceCube/reconstruction/transformer-encoder/examples/train_cnf/trained-models/allE_srt_213638022_213712969_large/flow_best.pth"
configpath="/home/axel/i3/LLP-at-IceCube/reconstruction/transformer-encoder/configs/model_high_energy_no_percentiles_large.yaml"
normpath="/home/axel/i3/LLP-at-IceCube/reconstruction/transformer-encoder/configs/normalization_args.yaml"
filenamestart="srt"

python extract_information.py \
 --topfolder "$topfolder" \
 --modelpath "$modelpath" \
 --flowpath "$flowpath" \
 --modelconfig "$configpath" \
 --filenamestart "$filenamestart" \
 --normpath "$normpath" \

cd run_scripts
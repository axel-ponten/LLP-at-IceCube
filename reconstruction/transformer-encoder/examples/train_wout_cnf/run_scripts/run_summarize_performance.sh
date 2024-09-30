
# topfolder="/home/axel/i3/i3-pq-conversion-files/srt_cleaned_DarkLeptonicScalar.mass-110.eps-3e-05.nevents-150000.0_ene_2000.0_15000.0_gap_100.0_240602.210981234/"
# modelpath=$PWD"/high_energy_set_srt_cleaned_models/model_no_cnf_epoch_75.pth"
# configpath="/home/axel/i3/LLP-at-IceCube/reconstruction/transformer-encoder/configs/model_no_cnf_high_energy_no_percentiles.yaml"
# normpath="/home/axel/i3/LLP-at-IceCube/reconstruction/transformer-encoder/configs/normalization_args.yaml"
# filenamestart="srt"
# nevents=4

cd ..

topfolder="/home/axel/i3/i3-pq-conversion-files/high_energy_srt_cleaned_combo_210981234_211644345_211828206/"
modelpath=$PWD"/trained-models/clipped_high_energy_srt_cleaned_210981234_211644345_211828206/model_epoch_80.pth"
configpath="/home/axel/i3/LLP-at-IceCube/reconstruction/transformer-encoder/configs/model_no_cnf_high_energy_no_percentiles.yaml"
normpath="/home/axel/i3/LLP-at-IceCube/reconstruction/transformer-encoder/configs/normalization_args.yaml"
filenamestart="srt"

python summarize_performance.py \
 --topfolder "$topfolder" \
 --modelpath "$modelpath" \
 --modelconfig "$configpath" \
 --filenamestart "$filenamestart" \
 --normpath "$normpath" \
#  --predicttrain \

cd run_scripts
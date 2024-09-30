cd ..

topfolder="/home/axel/i3/i3-pq-conversion-files/high_energy_srt_cleaned_combo_210981234_211644345_211828206"
normpath="/home/axel/i3/LLP-at-IceCube/reconstruction/transformer-encoder/configs/normalization_args.yaml"
epochs=200
batchsize=256
learningrate=0.0005
modelspath=$PWD"/trained-models/high_energy_srt_cleaned_combo_210981234_211644345_211828206_large/"
configpath="/home/axel/i3/LLP-at-IceCube/reconstruction/transformer-encoder/configs/model_high_energy_no_percentiles_large.yaml"
filenamestart="srt"

mkdir -p $modelspath

python -u train.py \
 --topfolder "$topfolder" \
 --normpath "$normpath" \
 --epochs "$epochs" \
 --batchsize "$batchsize" \
 --learningrate "$learningrate" \
 --modelspath "$modelspath" \
 --configpath "$configpath" \
 --filenamestart "$filenamestart" 2>&1 | tee $modelspath"/terminal_output.txt"

cd run_scripts
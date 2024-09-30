
topfolder="/home/axel/i3/i3-pq-conversion-files/high_energy_srt_cleaned_combo_210981234_211644345/"
normpath="/home/axel/i3/LLP-at-IceCube/reconstruction/transformer-encoder/configs/normalization_args.yaml"
epochs=200
batchsize=64
learningrate=0.0001
modelspath=$PWD"/high_energy_srt_combo_210981234_211644345"
configpath="/home/axel/i3/LLP-at-IceCube/reconstruction/transformer-encoder/configs/model_no_cnf_high_energy_no_percentiles.yaml"
filenamestart="srt"
do_plots=true

mkdir -p $modelspath

python -u train_no_cnf.py \
 --topfolder "$topfolder" \
 --normpath "$normpath" \
 --epochs "$epochs" \
 --batchsize "$batchsize" \
 --learningrate "$learningrate" \
 --modelspath "$modelspath" \
 --configpath "$configpath" \
 --filenamestart "$filenamestart" \
 --doplots "$do_plots" 2>&1 | tee $modelspath"/terminal_output.txt"

cd ..

topfolder="/home/axel/i3/i3-pq-conversion-files/uncleaned_allE_combo_213638022_213712969/"
normpath="/home/axel/i3/LLP-at-IceCube/reconstruction/transformer-encoder/configs/normalization_args.yaml"
modelspath=$PWD"/trained-models/allE_uncleaned_213638022_213712969_large/"
configpath="/home/axel/i3/LLP-at-IceCube/reconstruction/transformer-encoder/configs/model_high_energy_no_percentiles_large.yaml"
filenamestart="base"

epochs=300
batchsize=256
learningrate=0.0005

mkdir -p $modelspath

python -u train_new.py \
 --topfolder "$topfolder" \
 --normpath "$normpath" \
 --epochs "$epochs" \
 --batchsize "$batchsize" \
 --learningrate "$learningrate" \
 --modelspath "$modelspath" \
 --configpath "$configpath" \
 --filenamestart "$filenamestart" 2>&1 | tee $modelspath"/terminal_output.txt"
    
cd run_scripts

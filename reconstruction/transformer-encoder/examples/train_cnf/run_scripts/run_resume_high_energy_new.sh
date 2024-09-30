cd ..


topfolder="/home/axel/i3/i3-pq-conversion-files/high_energy_srt_cleaned_combo_210981234_211644345"
normpath="/home/axel/i3/LLP-at-IceCube/reconstruction/transformer-encoder/configs/normalization_args.yaml"
epochs=100
batchsize=256
# learningrate=0.0005
learningrate=0.00005
modelspath=$PWD"/trained-models/high_energy_srt_combo_210981234_211644345_gggggt"
configpath="/home/axel/i3/LLP-at-IceCube/reconstruction/transformer-encoder/configs/model_high_energy_no_percentiles.yaml"
filenamestart="srt"
lastepoch=150
# lastmodel=$modelspath"/model_epoch_"$lastepoch".pth"
# lastflow=$modelspath"/flow_epoch_"$lastepoch".pth"

lastmodel=$modelspath"/model_final.pth"
lastflow=$modelspath"/flow_final.pth"

mkdir -p $modelspath

python -u resume_new.py \
 --topfolder "$topfolder" \
 --normpath "$normpath" \
 --epochs "$epochs" \
 --batchsize "$batchsize" \
 --learningrate "$learningrate" \
 --modelspath "$modelspath" \
 --configpath "$configpath" \
 --laststatetransformer $lastmodel \
 --laststateflow $lastflow \
 --lastepoch $lastepoch \
 --filenamestart "$filenamestart" 2>&1 | tee $modelspath"/terminal_output_resumed.txt"
    
cd run_scripts

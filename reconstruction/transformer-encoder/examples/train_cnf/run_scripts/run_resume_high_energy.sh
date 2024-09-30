
topfolder="/home/axel/i3/i3-pq-conversion-files/high_energy_srt_cleaned_combo_210981234_211644345/"
normpath="/home/axel/i3/LLP-at-IceCube/reconstruction/transformer-encoder/configs/normalization_args.yaml"
epochs=50
batchsize=128
learningrate=0.0005
modelspath=$PWD"/trained-models/high_energy_srt_combo_210981234_211644345_only_t_reduceLRonPlateau/"
configpath="/home/axel/i3/LLP-at-IceCube/reconstruction/transformer-encoder/configs/model_high_energy_no_percentiles.yaml"
filenamestart="srt"
lastepoch=200

python -u resume_training.py \
 --topfolder "$topfolder" \
 --normpath "$normpath" \
 --epochs "$epochs" \
 --batchsize "$batchsize" \
 --learningrate "$learningrate" \
 --modelspath "$modelspath" \
 --configpath "$configpath" \
 --laststatetransformer $modelspath"/model_final.pth" \
 --laststateflow $modelspath"/flow_final.pth" \
 --lastepoch $lastepoch \
 --filenamestart "$filenamestart" 2>&1 | tee $modelspath"/terminal_output_resumed.txt"

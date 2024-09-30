cd ..

topfolder="/home/axel/i3/i3-pq-conversion-files/uncleaned_DarkLeptonicScalar.mass-110.eps-3e-05.nevents-500000.0_ene_200.0_15000.0_gap_50.0_240722.213638022/"
normpath="/home/axel/i3/LLP-at-IceCube/reconstruction/transformer-encoder/configs/normalization_args.yaml"
epochs=200
batchsize=256
learningrate=0.00005
modelspath=$PWD"/trained-models/all_energies_uncleaned_213638022_small"
configpath="/home/axel/i3/LLP-at-IceCube/reconstruction/transformer-encoder/configs/model_high_energy_no_percentiles.yaml"
filenamestart="base"

lastepoch=120
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

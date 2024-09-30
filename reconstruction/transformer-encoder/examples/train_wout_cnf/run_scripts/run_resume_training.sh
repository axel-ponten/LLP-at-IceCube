
topfolder="/home/axel/i3/i3-pq-conversion-files/srt_cleaned_DarkLeptonicScalar.mass-110.eps-3e-05.nevents-150000.0_ene_2000.0_15000.0_gap_100.0_240602.210981234/"
normpath="/home/axel/i3/LLP-at-IceCube/reconstruction/transformer-encoder/configs/normalization_args.yaml"
epochs=50
batchsize=128
learningrate=0.0001
modelspath=$PWD"/high_energy_set_srt_cleaned_large_models/"
laststate=$modelspath"model_no_cnf_final.pth"
lastepoch=75
configpath="/home/axel/i3/LLP-at-IceCube/reconstruction/transformer-encoder/configs/model_no_cnf_high_energy_no_percentiles_large.yaml"
filenamestart="srt"
do_plots=true


python resume_training_no_cnf.py \
 --topfolder "$topfolder" \
 --normpath "$normpath" \
 --epochs "$epochs" \
 --batchsize "$batchsize" \
 --learningrate "$learningrate" \
 --modelspath "$modelspath" \
 --configpath "$configpath" \
 --filenamestart "$filenamestart" \
 --doplots "$do_plots" \
 --laststate "$laststate" \
 --lastepoch "$lastepoch" 

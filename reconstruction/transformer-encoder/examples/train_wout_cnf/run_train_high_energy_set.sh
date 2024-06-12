
topfolder="/data/user/axelpo/converted-MC/DarkLeptonicScalar.mass-110.eps-3e-05.nevents-150000.0_ene_2000.0_15000.0_gap_100.0_240602.210981234/"
normpath="/data/user/axelpo/LLP-at-IceCube/reconstruction/configs/normalization_args.yaml"
epochs=400
batchsize=32
learningrate=0.0001
modelspath=$PWD"/high_energy_set_models/"
configpath="/data/user/axelpo/LLP-at-IceCube/reconstruction/configs/test_settings_no_cnf.yaml"
filenamestart="base"
do_plots=true


python train_no_cnf.py \
 --topfolder "$topfolder" \
 --normpath "$normpath" \
 --epochs "$epochs" \
 --batchsize "$batchsize" \
 --learningrate "$learningrate" \
 --modelspath "$modelspath" \
 --configpath "$configpath" \
 --filenamestart "$filenamestart" \
 --do_plots "$do_plots"

cd ..

filedir="/home/axel/i3/i3-pq-conversion-files/"

topfolder=($filedir"srt_cleaned_DarkLeptonicScalar.mass-110.eps-3e-05.nevents-500000.0_ene_200.0_15000.0_gap_50.0_240722.213638022/" \
$filedir"high_energy_uncleaned_210981234_211644345_211828206/" \
$filedir"high_energy_uncleaned_210981234_211644345_211828206/")

filenamestart=("srt" "base" "base")

modelpath=($PWD"/trained-models/all_energies_srt_213638022_small/model_epoch_240.pth" \
$PWD"/trained-models/clipped_high_energy_uncleaned_combo_210981234_211644345_211828206/model_epoch_60.pth" \
$PWD"/trained-models/clipped_high_energy_uncleaned_combo_210981234_211644345_211828206_large/model_epoch_90.pth")

flowpath=($PWD"/trained-models/all_energies_srt_213638022_small/flow_epoch_240.pth" \
$PWD"/trained-models/clipped_high_energy_uncleaned_combo_210981234_211644345_211828206/flow_epoch_60.pth" \
$PWD"/trained-models/clipped_high_energy_uncleaned_combo_210981234_211644345_211828206_large/flow_epoch_90.pth")

configpath=("/home/axel/i3/LLP-at-IceCube/reconstruction/transformer-encoder/configs/model_high_energy_no_percentiles.yaml" \
"/home/axel/i3/LLP-at-IceCube/reconstruction/transformer-encoder/configs/model_high_energy_no_percentiles.yaml" \
"/home/axel/i3/LLP-at-IceCube/reconstruction/transformer-encoder/configs/model_high_energy_no_percentiles_large.yaml")

normpath="/home/axel/i3/LLP-at-IceCube/reconstruction/transformer-encoder/configs/normalization_args.yaml"


len=${#topfolder[@]}

for ((i=0;i<len;i++))
do
    echo "New summary!"
    echo "${modelpath[i]}"
    echo "${flowpath[i]}"
    echo "${configpath[i]}"
    echo "${filenamestart[i]}"
    echo "$normpath"
    echo "${topfolder[i]}"

    python summarize_performance.py \
        --topfolder "${topfolder[i]}" \
        --modelpath "${modelpath[i]}" \
        --flowpath "${flowpath[i]}" \
        --modelconfig "${configpath[i]}" \
        --filenamestart "${filenamestart[i]}" \
        --normpath "$normpath"
done

cd run_scripts
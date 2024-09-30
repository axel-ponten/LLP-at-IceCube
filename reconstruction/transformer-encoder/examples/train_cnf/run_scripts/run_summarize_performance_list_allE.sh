cd ..

filedir="/home/axel/i3/i3-pq-conversion-files/"

topfolder=($filedir"srt_allE_combo_213638022_213712969/" \
$filedir"uncleaned_allE_combo_213638022_213712969/")

filenamestart=("srt" "base")

modelpath=($PWD"/trained-models/allE_srt_213638022_213712969_small/model_best.pth" \
$PWD"/trained-models/allE_uncleaned_213638022_213712969_small/model_best.pth")

flowpath=($PWD"/trained-models/allE_srt_213638022_213712969_small/flow_best.pth" \
$PWD"/trained-models/allE_uncleaned_213638022_213712969_small/flow_best.pth")

configpath=("/home/axel/i3/LLP-at-IceCube/reconstruction/transformer-encoder/configs/model_high_energy_no_percentiles.yaml" \
"/home/axel/i3/LLP-at-IceCube/reconstruction/transformer-encoder/configs/model_high_energy_no_percentiles.yaml")

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
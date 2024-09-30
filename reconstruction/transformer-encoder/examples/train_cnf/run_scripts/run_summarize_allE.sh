cd ..

filedir="/home/axel/i3/i3-pq-conversion-files/"

topfolder=($filedir"srt_allE_combo_213638022_213712969/")

filenamestart=("srt")

modelpath=($PWD"/trained-models/allE_srt_213638022_213712969_large/model_best.pth")

flowpath=($PWD"/trained-models/allE_srt_213638022_213712969_large/flow_best.pth")

configpath=("/home/axel/i3/LLP-at-IceCube/reconstruction/transformer-encoder/configs/model_high_energy_no_percentiles_large.yaml")

normpath="/home/axel/i3/LLP-at-IceCube/reconstruction/transformer-encoder/configs/normalization_args.yaml"

performancepath=($PWD"/trained-models/allE_srt_213638022_213712969_large/performance/performance.csv")

performancedir=($PWD"/trained-models/allE_srt_213638022_213712969_large/performance/")

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
    echo "${performancepath[i]}"
    echo "${performancedir[i]}"

    python summarize_performance.py \
        --topfolder "${topfolder[i]}" \
        --modelpath "${modelpath[i]}" \
        --flowpath "${flowpath[i]}" \
        --modelconfig "${configpath[i]}" \
        --filenamestart "${filenamestart[i]}" \
        --normpath "$normpath"

    echo "Visualize performance"
    python visualize_performance.py -i "${performancepath[i]}"

    echo "Reconstruct events"
    python plot_reconstructions.py \
        --topfolder "${topfolder[i]}" \
        --modelpath "${modelpath[i]}" \
        --flowpath "${flowpath[i]}" \
        --modelconfig "${configpath[i]}" \
        --filenamestart "${filenamestart[i]}" \
        --normpath "$normpath"

    echo "Reconstruct background events"
    python plot_events_corner_bkg.py \
        --topfolder "${topfolder[i]}" \
        --modelpath "${modelpath[i]}" \
        --flowpath "${flowpath[i]}" \
        --modelconfig "${configpath[i]}" \
        --filenamestart "${filenamestart[i]}" \
        --normpath "$normpath"

    # create summary pdf document
    echo "Create summary pdf"
    python create_summary_pdf.py \
        -p "${performancedir[i]}"
done

cd run_scripts
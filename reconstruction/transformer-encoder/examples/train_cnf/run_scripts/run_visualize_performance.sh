cd ../

filename="performance/performance.csv"

file1=$PWD/trained-models/allE_srt_213638022_213712969_small/$filename
file2=$PWD/trained-models/allE_uncleaned_213638022_213712969_small/$filename

python visualize_performance.py -i $file1,$file2

cd run_scripts/
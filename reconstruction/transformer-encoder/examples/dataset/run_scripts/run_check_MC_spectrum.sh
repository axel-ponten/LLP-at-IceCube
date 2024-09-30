cd ../

parentfolder="/home/axel/i3/i3-pq-conversion-files/"

# topfolder=$parentfolder"srt_allE_combo_213638022_213712969/"
# python check_MC_spectrum.py --topfolder $topfolder --save

# topfolder=$parentfolder"high_energy_srt_cleaned_210981234_211644345_211828206/"
# python check_MC_spectrum.py --topfolder $topfolder --save

# topfolder=$parentfolder"high_energy_uncleaned_210981234_211644345_211828206/"
# python check_MC_spectrum.py --topfolder $topfolder --save

# topfolder=$parentfolder"srt_allE_combo_213638022_213712969/"
# python check_MC_spectrum.py --topfolder $topfolder --save

# topfolder=$parentfolder"srt_cleaned_IC86.2020_corsika.020904.198000.i3.zst/"
# python check_MC_spectrum.py --topfolder $topfolder --save

# topfolder=$parentfolder"uncleaned_allE_combo_213638022_213712969/"
# python check_MC_spectrum.py --topfolder $topfolder --save


topfolder=$parentfolder"uncleaned_allE_combo_213638022_213712969/"
python check_MC_spectrum.py --topfolder $topfolder --save


cd run_scripts
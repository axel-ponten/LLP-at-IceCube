import argparse
from converter import Converter
import pandas as pd

if __name__ == "__main__":
    # read in parser
    parser = argparse.ArgumentParser('convert_DLS')

    # gcd
    gcdfile = "/data/sim/sim-new/downloads/GCD/GeoCalibDetectorStatus_2021.Run135903.T00S1.Pass2_V1b_Snow211115.i3.gz"

    # which files to convert? list of paths
    # filenames = ["test.i3.gz"]
    filenames = ["L2test2.i3.gz"]
    #df_filenames = pd.read_csv("i3files.csv")
    #filenames = list(df_filenames)
    # print("filenames", filenames)
    
    target_folder = "/data/user/axelpo/conversion_testing_ground/"
    
    # run conversion
    Converter(
            filenames,
            target_folder,
            encoding_type="mod_harnisch",
            pulse_series_name="InIcePulses",
            gcdfile=gcdfile,
            num_events_per_file=300)

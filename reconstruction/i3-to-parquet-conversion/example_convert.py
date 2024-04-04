import argparse
from converter import Converter
import pandas as pd

if __name__ == "__main__":
    # read in parser
    parser = argparse.ArgumentParser('convert_DLS')

    parser.add_argument("-t", "--target-folder", action="store",
            type=str, default="/data/user/axelpo/conversion_testing_ground/", dest="target-folder",
            help="Directory where to store parquet output. Include backslash.")
    parser.add_argument("-n", "--num-events-per-file", action="store",
            type=int, default=1000, dest="num-events-per-file",
            help="Number of events per parquet file.")
    parser.add_argument("-g", "--gcdfile", action="store",
            type=str, default="/data/sim/sim-new/downloads/GCD/GeoCalibDetectorStatus_2021.Run135903.T00S1.Pass2_V1b_Snow211115.i3.gz", dest="gcdfile",
            help="GCD file.")
    parser.add_argument("-e", "--encoding", action="store",
        type=str, default="mod_harnisch", dest="encoding",
        help="Encoding type from feature config to be passed to icecube.ml_suite.EventFeatureFactory")
    parser.add_argument("--no-llp", action="store_false",
        default=True, dest="is-llp",
        help="Is the dataset an LLP MC simulation?")
    
    params = vars(parser.parse_args())  # dict()

    # which files to convert? list of paths
    filenames = ["L2test2.i3.gz"]
    
    # run conversion
    Converter(
            filenames,
            target_folder = params["target-folder"],
            encoding_type="mod_harnisch",
            pulse_series_name="InIcePulses",
            gcdfile=params["gcdfile"],
            num_events_per_file=params["num-events-per-file"],
            num_per_row_group=100,
            is_llp=params["is-llp"],
    )

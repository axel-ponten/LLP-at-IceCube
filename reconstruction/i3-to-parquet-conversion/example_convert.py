import argparse
from converter import Converter
import os

if __name__ == "__main__":
    # read in parser
    parser = argparse.ArgumentParser('convert_DLS')

    parser.add_argument("-t", "--target-folder", action="store",
            type=str, required=True, dest="target-folder",
            help="Directory where to store parquet output. Include backslash.")
    parser.add_argument("-i", "--input-file", action="store",
            type=str, required=True, dest="input-file",
            help="Input i3 file(s). Separate multiple files with commas.")
    parser.add_argument("-n", "--num-events-per-file", action="store",
            type=int, default=1000, dest="num-events-per-file",
            help="Number of events per parquet file.")
    parser.add_argument("-r", "--row-group-number", action="store",
            type=int, default=100, dest="row-group-number",
            help="Number of events per row group in parquet file.")
    parser.add_argument("-g", "--gcdfile", action="store",
            type=str, default="/data/sim/sim-new/downloads/GCD/GeoCalibDetectorStatus_2021.Run135903.T00S1.Pass2_V1b_Snow211115.i3.gz", dest="gcdfile",
            help="GCD file.")
    parser.add_argument("-e", "--encoding", action="store",
        type=str, default="llp_test_config", dest="encoding",
        help="Encoding type from feature config to be passed to icecube.ml_suite.EventFeatureFactory")
    parser.add_argument("--no-llp", action="store_false",
        default=True, dest="is-llp",
        help="Is the dataset an LLP MC simulation?")
    parser.add_argument("-p", "--pulse-series-name", action="store",
            type=str, default="InIcePulses", dest="pulse-series-name",
            help="Name of pulse series to use for pulse extraction.")
    parser.add_argument("-s", "--sub-event-stream", action="store",
            default=None, dest="sub-event-stream",
            help="Sub event stream. None (default) for DAQ frames, 'InIceSplit' for P frame.")
    
    params = vars(parser.parse_args())  # dict()

    # add trailing slash to target folder if not present
    if params["target-folder"][-1] != "/":
        params["target-folder"] += "/"

    # create target folder if it does not exist        
    try:
        os.makedirs(params["target-folder"])
        print("Created target folder since it did not exist.")
    except FileExistsError:
        pass
    
    # save info about the conversion to yaml file
    with open(params["target-folder"] + "conversion_info.yaml", "w") as f:
        import yaml
        import feature_configs
        encoding = feature_configs.feature_configs[params["encoding"]]
        params["encoding-string"] = encoding
        yaml.dump(params, f)

    # split input files. If only one file, split will return a list with one element
    filenames = params["input-file"].split(",")

    # run conversion
    Converter(
            filenames,
            target_folder = params["target-folder"],
            encoding_type=params["encoding"],
            pulse_series_name=params["pulse-series-name"],
            sub_event_stream=params["sub-event-stream"],
            gcdfile=params["gcdfile"],
            num_events_per_file=params["num-events-per-file"],
            num_per_row_group=params["row-group-number"],
            is_llp=params["is-llp"],
    )

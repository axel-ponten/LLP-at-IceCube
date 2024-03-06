import sys
sys.path.append("..")

import glob
import yaml
import argparse
from datetime import strftime, datetime
import icecube
from icecube import icetray, dataio
from icecube.icetray import I3Tray

from I3LLPProbabilityCalculator import *
from llpestimation import LLPModel, LLPEstimator, LLPMedium, LLPProductionCrossSection
from estimation_utilities import *

class Calculate:
    def __init__(self, args):
        self.model = args.model
        self.outdir = args.outdir
        self.config = args.config
        
    def parseConfig(self):    
        with open('config.yml', 'r') as f:
            yaml_config = self.config.read()

        self.parsed_config = yaml.safe_load(yaml_config)
        self.model_package = self.parsed_config["models"][self.model]
    
    def parseFiles(self):
        # infiles
        filelist = list(glob.glob(self.parsed_config['topdir']),)
        n_files = 1 # how many files to use?
        self.filelist = filelist[0:n_files]
        print("Number of CORSIKA files used:", len(self.filelist))
        self.gcdfile = self.parsed_config['gcd']

    def countEvents(self):
        # create LLP models
        masses   = self.model_package["masses"]
        epsilons = self.model_package["epsilons"]["nominal"]
        names    = ["DLS" for _ in masses]
        table_paths = generate_DLS_WW_oxygen_paths(masses, folder = "../cross_section_tables/")
        DLS_models = generate_DLSModels(masses, epsilons, names, table_paths)

        min_gap = self.model_package["mingap"] # minimum detectable LLP gap
        # create LLPEstimator
        DLS_estimator = LLPEstimator(DLS_models, min_gap)
        return DLS_estimator
    
    def print_LLPInfo(frame):
        llp_map = dict(frame["LLPProbabilities"])
        print(llp_map)
        return True
    
    def createTray(self):
        DLS_estimator = Calculate.countEvents()
        # detector parameters
        n_steps = 50

        ########## Run I3Tray ##########
        tray = I3Tray()

        tray.Add("I3Reader", FileNameList=self.filelist)
        tray.Add(I3LLPProbabilityCalculator,
                GCDFile = self.gcdfile,
                llp_estimator = DLS_estimator,
                n_steps = n_steps
        )
        #tray.Add(print_LLPInfo, Streams=[icecube.icetray.I3Frame.DAQ])
        date = datetime.now().strftime('%m%d%Y')
        tray.Add("I3Writer", filename=f"{self.outdir}/I3ProbabilityCalculator_{self.model}_{date}_output.i3.gz")
        tray.Execute()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model',
                        help='model used', defaut='DarkLeptonicScalar')
    parser.add_argument('-o','--outdir',
                        help='output di',
                        default='out/')
    parser.add_argument('-c','--config',
                        default='config.yaml',
                        help='what configuration file are you using?')
        
    args = parser.parse_args()
    passArgs = Calculate(args)
    
    Calculate.parseConfig()
    Calculate.parseFiles()
    Calculate.createTray()
    

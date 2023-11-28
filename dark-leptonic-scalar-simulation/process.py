#!/usr/bin/env python3
### The main L2 processing script ###

import sys
import subprocess

from icecube.icetray import I3Tray
from icecube import icetray, dataclasses, dataio 
from icecube.icetray import I3PacketModule, I3Units

from icecube.filterscripts.offlineL2.level2_all_filters import OfflineFilter
from icecube.filterscripts.offlineL2 import SpecialWriter

def make_parser():
    """Make the argument parser"""
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-s", "--simulation", action="store_true",
        default=False, dest="mc",
        help="Mark as simulation (MC)")
    parser.add_argument("-i", "--input", action="store",
        type=str, default="", dest="infile",
        help="Input i3 file(s)  (use comma separated list for multiple files)")
    parser.add_argument("-g", "--gcd", action="store",
        type=str, default="", dest="gcdfile",
        help="GCD file for input i3 file")
    parser.add_argument("-o", "--output", action="store",
        type=str, default="", dest="outfile",
        help="Output i3 file")
    parser.add_argument("-n", "--num", action="store",
        type=int, default=-1, dest="num",
        help="Number of frames to process")
    parser.add_argument("--dstfile", action="store",
        type=str, default=None, dest="dstfile",
        help="DST root file (should be .root)")
    parser.add_argument("--gapsfile", action="store",
        type=str, default=None, dest="gapsfile",
        help="gaps text file (should be .txt)")
    parser.add_argument("--icetopoutput", action="store",
        type=str, default=None, dest="icetopoutput",
        help="Output IceTop file")
    parser.add_argument("--eheoutput", action="store",
        type=str, default=None, dest="eheoutput",
        help="Output EHE i3 file")
    parser.add_argument("--slopoutput", action="store",
        type=str, default=None, dest="slopoutput",
        help="Output SLOP file")
    parser.add_argument("--rootoutput", action="store",
        type=str, default=None, dest="rootoutput",
        help="Output root file")
    parser.add_argument("--photonicsdir", action="store",
        type=str, default=None,
        dest="photonicsdir", help="Directory with photonics tables")
    parser.add_argument("--log-level",
        type=str, default="WARN", dest="LOG_LEVEL",
        help="Sets the logging level (ERROR, WARN, INFO, DEBUG, TRACE)")
    parser.add_argument("--log-filename",
        type=str, default=None, dest="logfn",
        help="If set logging is redirected to the specified file.")

    return parser

def main(options, stats={}):
    """The main L2 processing script"""
    tray = I3Tray()

    log_levels = {"error" : icetray.I3LogLevel.LOG_ERROR,
                  "warn" : icetray.I3LogLevel.LOG_WARN,
                  "info" : icetray.I3LogLevel.LOG_INFO,
                  "debug" : icetray.I3LogLevel.LOG_DEBUG,
                  "trace" : icetray.I3LogLevel.LOG_TRACE}

    if options['LOG_LEVEL'].lower() in log_levels.keys():
        icetray.set_log_level(log_levels[options['LOG_LEVEL'].lower()])
    else:
        print("log level option %s not recognized.")
        print("Options are ERROR, WARN, INFO, DEBUG, and TRACE.")
        print("Sticking with default of WARN.")
        icetray.set_log_level(icetray.I3LogLevel.LOG_WARN)

    if options['logfn']:
        import os
        pid = os.getpid()
        icetray.logging.rotating_files("%s.%d" % (options['logfn'], pid))
        
    # make list of input files from GCD and infile
    if isinstance(options['infile'],list):
        infiles = [options['gcdfile']]
        infiles.extend(options['infile'])
    else:
        infiles = [options['gcdfile'], options['infile']]
    print('infiles: ',infiles)
    
    # test access to input and output files
    for f in infiles:
        import os
        if not os.access(f,os.R_OK):
            raise Exception('Cannot read from %s'%f)
    def test_write(f):
        import os
        if f:
            try:
                open(f,'w')
            except IOError:
                raise Exception('Cannot write to %s'%f)
            finally:
                os.remove(f)
    test_write(options['outfile'])
    test_write(options['dstfile'])
    test_write(options['gapsfile'])
    test_write(options['icetopoutput'])
    test_write(options['eheoutput'])
    test_write(options['slopoutput'])
    test_write(options['rootoutput'])

    # read input files
    tray.Add( dataio.I3Reader, "Reader",
        Filenamelist = infiles
    )

    tray.AddSegment(OfflineFilter, "OfflineFilter",
        dstfile=options['dstfile'],
        mc=options['mc'],
        doNotQify=options['mc'],
        photonicsdir=options['photonicsdir']
        )

    ###################################################################
    ########### WRITE STUFF                  ##########################
    ###################################################################
    
    # clean up special case for bz2 files to get around empty file bug
    bzip2_files = []
    if options['outfile'] and options['outfile'].endswith('.bz2'):
        options['outfile'] = options['outfile'][:-4]
        bzip2_files.append(options['outfile'])
    if options['dstfile'] and options['dstfile'].endswith('.bz2'):
        options['dstfile'] = options['dstfile'][:-4]
        bzip2_files.append(options['dstfile'])
    if options['icetopoutput'] and options['icetopoutput'].endswith('.bz2'):
        options['icetopoutput'] = options['icetopoutput'][:-4]
        bzip2_files.append(options['icetopoutput'])
    if options['eheoutput'] and options['eheoutput'].endswith('.bz2'):
        options['eheoutput'] = options['eheoutput'][:-4]
        bzip2_files.append(options['eheoutput'])
    if options['slopoutput'] and options['slopoutput'].endswith('.bz2'):
        options['slopoutput'] = options['slopoutput'][:-4]
        bzip2_files.append(options['slopoutput'])
    
    tray.AddModule( "I3Writer", "EventWriter" ,
        Filename = options['outfile'],
        Streams = [icetray.I3Frame.DAQ, icetray.I3Frame.Physics,
                   icetray.I3Frame.TrayInfo, icetray.I3Frame.Simulation,
                   icetray.I3Frame.Stream('M')], # added M-frame
        DropOrphanStreams = [icetray.I3Frame.DAQ],
    )

    # special outputs
    if options['eheoutput']:
        tray.AddSegment(SpecialWriter.EHEWriter, "write_ehe",
            Filename=options['eheoutput']
        )
    if options['slopoutput']:
        tray.AddSegment(SpecialWriter.SLOPWriter, "write_slop",
            Filename=options['slopoutput']
        )
    if options['rootoutput']:
        tray.AddSegment(SpecialWriter.RootWriter, "write_root",
            Filename = options['rootoutput']
        )
    if options['gapsfile']:
        tray.AddSegment(SpecialWriter.GapsWriter, "write_gaps",
            Filename = options['gapsfile'],
            MinGapTime = 1
        )
    if options['icetopoutput']:
        # this needs to be the last output
        tray.AddModule('Delete','deleteicetopextrakeys',
            keys=['InIceRawData','CleanInIceRawData']
        )
        tray.AddSegment(SpecialWriter.IceTopWriter, "write_icetop",
            Filename=options['icetopoutput']
        )

    

    # make it go
    if options['num'] >= 0:
        tray.Execute(options['num'])
    else:
        tray.Execute()

    
   
    # print more CPU usage info. than speicifed by default
    tray.PrintUsage(fraction=1.0) 
    for entry in tray.Usage():
        stats[entry.key()] = entry.data().usertime

    if bzip2_files:
        import os
        # now do bzip2
        if os.path.exists('/usr/bin/bzip2'):
            subprocess.check_call(['/usr/bin/bzip2', '-f']+bzip2_files)   
        elif os.path.exists('/bin/bzip2'):
            subprocess.check_call(['/bin/bzip2', '-f']+bzip2_files)   
        else:
            raise Exception('Cannot find bzip2')
 
    # clean up forcefully in case we're running this in a loop
    del tray



### iceprod stuff ###
try:
    from iceprod.modules import ipmodule
except ImportError as e:
    icetray.logging.log_warn('Module iceprod.modules not found. Will not define IceProd Class')
else:
    Level2Filter = ipmodule.FromOptionParser(make_parser(),main)
### end iceprod stuff ###

if __name__ == '__main__':
    # run as script from the command line
    # get parsed args
    parser = make_parser()
    args = parser.parse_args()

    # convert to dictionary
    opts = vars(args)
    opts['infile'] = opts['infile'].split(',')

    # call main function
    main(opts)

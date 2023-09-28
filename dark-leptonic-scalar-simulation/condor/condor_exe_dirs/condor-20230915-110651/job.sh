#!/bin/bash
set -e
printf "Start time: "; /bin/date
printf "Job is running on node: "; /bin/hostname
printf "Job is running in directory: "; /bin/pwd

eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.2.1/setup.sh)
/data/user/axelpo/i3/icetray-axel/build/env-shell.sh python $@
echo "Job complete!"

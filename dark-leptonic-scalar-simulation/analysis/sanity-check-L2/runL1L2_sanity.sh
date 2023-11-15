L1SCRIPT=/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_7_x86_64/metaprojects/icetray/v1.8.2/filterscripts/resources/scripts/SimulationFiltering.py
L2SCRIPT=/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_7_x86_64/metaprojects/icetray/v1.8.2/filterscripts/resources/scripts/offlineL2/process.py
CURRENTDIR=${PWD}
INFILE=${CURRENTDIR}/20904_examples/IC86.2020_corsika.020904.000001.i3.zst

python ${L1SCRIPT} -g $GCDFILE -o ${CURRENTDIR}/L1_sanity.i3.gz -i ${INFILE}
python ${L2SCRIPT} -s -g $GCDFILE -o ${CURRENTDIR}/L2_sanity.i3.gz -i ${CURRENTDIR}/L1_sanity.i3.gz


export JOBFILE=job.sh

# simulation parameters
export NJOBS=500
export NEVENTS=500
export MASS=115
export EPS=5e-6
export BIAS=1e9
export MODEL=DarkLeptonicScalar
export MINLENGTH=50
export DATASETID=7777
export FLUXMODEL=Hoerandel5_atmod12_SIBYLL
export FLUXGAMMA=2
export FLUXOFFSET=700
export FLUXEMIN=1e2
export FLUXEMAX=2e5

export CURRENTDATE=`date +%y%m%d`

#create exepath to avoid condor incident
export EXEDIR="condor_exe_dirs/condor-$(date +%Y%m%d-%H%M%S)"
mkdir $EXEDIR
cp $JOBFILE $EXEDIR

#skript used for condor
export CONDORSCRIPT=$(pwd)"/condor_submit_template/FullSimulationLLP_template_v2.sub"

#transform condor
sed -e 's/<njobs>/'$NJOBS'/g' \
    -e 's/<nevents>/'$NEVENTS'/g' \
    -e 's/<mass>/'$MASS'/g' \
    -e 's/<eps>/'$EPS'/g' \
    -e 's/<model>/'$MODEL'/g' \
    -e 's/<bias>/'$BIAS'/g' \
    -e 's/<datasetid>/'$DATASETID'/g' \
    -e 's/<currentdate>/'$CURRENTDATE'/g' \
    -e 's/<minlength>/'$MINLENGTH'/g' \
    -e 's/<fluxmodel>/'$FLUXMODEL'/g' \
    -e 's/<fluxgamma>/'$FLUXGAMMA'/g' \
    -e 's/<fluxoffset>/'$FLUXOFFSET'/g' \
    -e 's/<fluxemin>/'$FLUXEMIN'/g' \
    -e 's/<fluxemax>/'$FLUXEMAX'/g' \
    $CONDORSCRIPT > "$EXEDIR/condor.submit";


#call condor skript
echo "CALLING CONDOR SERVICE:"
cd $EXEDIR
condor_submit condor.submit

#back to original directory
echo "back to the original directory"
cd -

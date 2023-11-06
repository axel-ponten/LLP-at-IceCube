#!/usr/bin/env python3

"""
Example millipede muon energy loss fits.  The first fits the
loss pattern as stochastic losses (e.g. from a high-energy
muon), and the second as continuous losses.
input: offline reconstructed .i3 file(s)
"""

from icecube.icetray import I3Tray
import sys,os
from icecube import icetray, dataio, dataclasses, photonics_service, millipede

if len(sys.argv) < 3:
	print('Usage: %s output.i3 input1.i3 [input2.i3] ...' % sys.argv[0])
	sys.exit(1)

def extract_inice_quantities(frame):
    mctree=frame["I3MCTree"]
    maxenergy = 0
    for p in frame["I3MCTree_preMuonProp"].children(frame["I3MCTree_preMuonProp"].get_head()):
        if p.energy > maxenergy:
            maxparticle = p
            maxenergy = maxparticle.energy
    frame["true_mc_dir"] = maxparticle
    #frame["most_energetic_cascade"]=dataclasses.get_most_energetic_inice_cascade(mctree)
    #frame["most_energetic_track"]=dataclasses.get_most_energetic_muon(mctree)
    #frame["most_energetic_neutrino"]=dataclasses.get_most_energetic_neutrino(mctree)
    
    return True
    
files = sys.argv[2:]

table_base = os.path.expandvars('$I3_DATA/photon-tables/splines/emu_%s.fits')
muon_service = photonics_service.I3PhotoSplineService(table_base % 'abs', table_base % 'prob', timingSigma=0)
table_base = os.path.expandvars('$I3_DATA/photon-tables/splines/ems_spice1_z20_a10.%s.fits')
cascade_service = photonics_service.I3PhotoSplineService(table_base % 'abs', table_base % 'prob', timingSigma=0)

tray = I3Tray()
tray.AddModule('I3Reader', 'reader', FilenameList=files)
tray.Add(extract_inice_quantities, Streams=[icetray.I3Frame.Physics])
tray.AddModule('MuMillipede', 'millipede_highenergy',
    MuonPhotonicsService=muon_service, CascadePhotonicsService=cascade_service,
    PhotonsPerBin=15, MuonRegularization=0, ShowerRegularization=0,
    MuonSpacing=0, ShowerSpacing=10, SeedTrack='true_mc_dir',
    Output='MillipedeHighEnergy', Pulses='SplitInIcePulses')

tray.AddModule('MuMillipede', 'millipede_lowenergy',
    MuonPhotonicsService=muon_service, CascadePhotonicsService=cascade_service,
    PhotonsPerBin=10, MuonRegularization=2, ShowerRegularization=0,
    MuonSpacing=15, ShowerSpacing=0, SeedTrack='true_mc_dir',
    Output='MillipedeLowEnergy', Pulses='SplitInIcePulses')

tray.AddModule('I3Writer', 'writer', filename=sys.argv[1])

tray.Execute()



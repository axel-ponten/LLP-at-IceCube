import icecube
from icecube import dataclasses, dataio, icetray, simclasses, MuonGun, hdfwriter
from I3Tray import I3Tray
import glob

filename  = "/data/user/axelpo/LLP-at-IceCube/dark-leptonic-scalar-simulation/analysis/sanity-check-L2/L1_sanity.i3.gz"

tray = I3Tray()

tray.Add("I3Reader", FileName=filename)
tray.Add(
    hdfwriter.I3SimHDFWriter,
    keys=["PolyplopiaPrimary", "CorsikaWeightMap"],
    output=filename[:-6]+".hdf5",
)

tray.Execute()

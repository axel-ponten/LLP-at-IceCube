# LLP-at-IceCube

Repository for the Long Lived Particles (LLP) at IceCube analysis. Sub-folders contain separate README.md.

Contact:

* Axel Pontén, PhD student at Uppsala University, axel.ponten@physics.uu.se
* Victoria Parrish, PostDoc at Michigan State University, vparrish@icecube.wisc.edu
* Carlos Perez de los Heros, Professor at Uppsala University, cph@physics.uu.se

## Background
This project is part of a beyond standard model analysis at the IceCube neutrino observatory, which is a cube kilometer large neutrino detector buried 2 km deep in the antarctic ice (https://icecube.wisc.edu/). The detector is built to measure rare cosmic neutrino interactions a few times a year, but also detects the abundant atmospheric muon background at a rate of 3 kHz. The analysis searches for signatures of a hypothesized class of exotic particles called long lived particles (LLPs) in the background atmospheric muon flux. The LLPs could potentially be produced in a bremstrahlung-like process where the muon scatters of an oxygen or hydrogen in the ice. The LLP would then decay after some time, hopefully inside the detector for it to be detectable.

IceCube measures particles through their Cherenkov radiation in the ice. LLPs are dark and do not emit any Cherenkov light. Their production and subsequent decay would therefore leave a gap in the muon track where there is no light emitted. As such, we need both the production and decay vertex of the LLP to be inside the detector volume in order for the event to be detectable.

## dark-leptonic-scalar-simulation
dark-leptonic-scalar-simulation folder contains scripts for simulating LLP events and analyzing the produced MC set.

## sensitivity-estimation
sensitivity-estimation contains scripts to estimate the expected production of LLPs in IceCube. Since the atmospheric muon spectrum spans many energies, zeniths and intersections in the detector (entry and exit point), we estimate the event-by-event probability of LLP production and weight this over a complete spectrum. The main workhorse for this is the custom package *llpestimation*. For the spectrum we use CORSIKA-in-ice simulation which we pass through and send each event to the *LLPEstimator* from the *llpestimation* package.

The folder *mathematica* contains mathematica notebooks to compute interpolation tables for total cross section of some LLP model. @TODO: currently only dark leptonic scalar (DLS) implemented.

The *grid* folder contains scripts to estimate production over many masses and epsilons of the LLP, as is common in LLP analyses.

Note that no reconstruction or systematic effects beyond detector geometry are considered in the estimation, purely length and energy of muons inside the detector.

## reconstruction

reconstruction contains scripts that reconstruct the LLP track gap from pulses using ML techniques. @TODO: so far only scripts to convert .i3 files to .pq files are there.

## background-study
background-study contains scripts to estimate stopping muon + collinear neutrino background. @TODO: This is incomplete and needs additional work.

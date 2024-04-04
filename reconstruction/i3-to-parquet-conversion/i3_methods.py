import numpy
import glob
import os
import yaml
import time

from icecube import icetray, dataio, dataclasses, MuonGun
from icecube.ml_suite import EventFeatureFactory
from icecube.sim_services import label_events


#Function to read the GCD file and make the extruded polygon which
#defines the edge of the in-ice array
def MakeSurface(gcdName, padding):
    file = dataio.I3File(gcdName, "r")
    frame = file.pop_frame()
    while not "I3Geometry" in frame:
        frame = file.pop_frame()
    geometry = frame["I3Geometry"]
    xyList = []
    zmax = -1e100
    zmin = 1e100
    step = int(len(geometry.omgeo.keys())/10)
    print("Loading the DOM locations from the GCD file")
    for i, key in enumerate(geometry.omgeo.keys()):
        if i % step == 0:
            print( "{0}/{1} = {2}%".format(i,len(geometry.omgeo.keys()), int(round(i/len(geometry.omgeo.keys())*100))))
            
        if key.om in [61, 62, 63, 64] and key.string <= 81: #Remove IT...
            continue

        pos = geometry.omgeo[key].position

        if pos.z > 1500:
            continue

        xyList.append(pos)
        i+=1
    
    return MuonGun.ExtrudedPolygon(xyList, padding)

def extract_geometry_n_rdes(gcd_file):
    """
    Extracts geometry from gcd file and returns dict geometry.
    """
    dict_geo=dict()

    f=dataio.I3File(gcd_file)
    
    fr=f.pop_frame()
    while(fr.Stop != icetray.I3Frame.Geometry):
        fr=f.pop_frame()

    geo=fr["I3Geometry"].omgeo

    for k in geo.keys():
        if(k.om<=60):
            dict_geo[k]=[geo[k].position.x,geo[k].position.y,geo[k].position.z]

    ## extract RDE s

    c_frame=f.pop_frame()

    cal=c_frame["I3Calibration"]
        
    rde_dict=dict()

    for om_key in cal.dom_cal.keys():
        rde_dict[om_key]=cal.dom_cal[om_key].relative_dom_eff
    f.close()

    return fr["I3Geometry"], dict_geo, rde_dict

def weighted_quantile(data, weights, perc):
    """
    perc : percentile in [0-1]!
    """
    ix = numpy.argsort(data)
    data = data[ix] # sort data
    weights = weights[ix] # sort weights
    cdf = (numpy.cumsum(weights) - 0.5 * weights) / numpy.sum(weights) # 'like' a CDF function
    return numpy.interp(perc, cdf, data)

def obtain_llp_data(frame, surface=None):
    """ Return frame["LLPInfo"] information
    
    return (production, decay, direction,
            gap_length, fractional_energy, llp_energy,
            decay_asymmetry)
    """
    if "LLPInfo" not in frame:
        raise ValueError("LLPInfo not in frame")
    # production and compute decay point
    direction  = dataclasses.I3Direction(frame["LLPInfo"]["zenith"],
                                         frame["LLPInfo"]["azimuth"])
    production = dataclasses.I3Position(frame["LLPInfo"]["prod_x"],
                                        frame["LLPInfo"]["prod_y"],
                                        frame["LLPInfo"]["prod_z"])
    decay = production + \
            dataclasses.I3Position(frame["LLPInfo"]["length"],
                                   direction.theta,
                                   direction.phi,
                                   dataclasses.I3Position.sph)
    # if surface is not None:
    #     # negative value means intersection behind point, positive means in front
    #     prod_intersection  = surface.intersection(production, direction)
    #     decay_intersection = surface.intersection(decay, direction)

    gap_length        = frame["LLPInfo"]["length"]
    fractional_energy = frame["LLPInfo"]["fractional_energy"]
    llp_energy        = frame["LLPInfo"]["llp_energy"]
    decay_asymmetry   = frame["LLPInfo"]["decay_asymmetry"]
    
    return (production, decay, direction,
            gap_length, fractional_energy, llp_energy,
            decay_asymmetry)

def obtain_encoded_data(frame, 
                        pulse_series_name, 
                        geometry,
                        rde_dict,
                        feature_extractor=None,
                        feature_indices=None,):
    
    # get pulses
    pulses = frame[pulse_series_name]

    if(type(pulses)==dataclasses.I3RecoPulseSeriesMapMask):
        pulses=pulses.apply(frame)

    if(len(pulses)==0):
        print(pulse_series_name, " has no pulses ....")
        print(frame["I3EventHeader"])

        raise Exception(pulse_series_name, " has no pulses ....", frame[pulse_series_name])

    string_id_list=[]
    om_id_list=[]
    module_type_list=[]

    ## see if we can load a yaml feature config file
    assert(feature_extractor is not None)
    # map of OMKey -> feature vector
    res=feature_extractor.get_feature_map(frame)

    feature_vec=[] # we will modify the feature_extractor feature vec

    total_charges=[]
    weighted_positions=[]

    median_times=[]

    charge_weighted_time_mean=0

    # for each dom
    for k in res.keys():
        # add xyz of DOM
        feature_vec.append(numpy.concatenate( [numpy.array(geometry[k]), numpy.array(res[k])])[None,...])

        string_id_list.append(k.string)
        om_id_list.append(k.om)
        # classification of deepcore DOMs, 0 for normal, 1 for deepcore
        if(rde_dict[k]==1.0):
            module_type_list.append(0)
        elif(rde_dict[k]>1.3): # 130% relative dom efficiency
            module_type_list.append(1)
        else:
            raise Exception("WEIRD RDE?!", k, rde_dict[k])

        ##########################################################

        total_charges.append(feature_vec[-1][0, feature_indices["log_total_charge"]])

        ####

        this_time_list=[p.time for p in pulses[k]]
        this_times=numpy.array(this_time_list)
    
        this_charge_list=[p.charge for p in pulses[k]]
        this_charges=numpy.array(this_charge_list)

        
        assert(sum(this_charges)==total_charges[-1]), ("tot charge mismatch ", k, "sum python: ", sum(this_charges), " sum c++ feature extractor ", total_charges[-1])

        charge_weighted_time_mean+=(this_charges*this_times).sum()

        # calc median

        time_median=weighted_quantile(this_times, this_charges, numpy.array([0.5]))[0]
        median_times.append(time_median)

        
        #feature_vec[-1][0, feature_indices["log_total_charge"]]=numpy.log(feature_vec[-1][0, feature_indices["log_total_charge"]]+1.0)
        weighted_positions.append(sum(this_charges)*numpy.array(geometry[k]))
        ####################################

    overall_charge_in_event=sum(total_charges)
    charge_weighted_time_mean/=overall_charge_in_event

    mean_cog_pos=numpy.sum(weighted_positions,axis=0)/sum(total_charges)
    string_id_list=numpy.array(string_id_list, dtype=numpy.int64)
    om_id_list=numpy.array(om_id_list, dtype=numpy.int64)
    module_type_list=numpy.array(module_type_list, dtype=numpy.int64)

    prepared_encoded_data=numpy.concatenate(feature_vec)
    
    # log(totcharge + 1) cus charge can be 0 and charge spans orders of magnitude
    prepared_encoded_data[:, feature_indices["log_total_charge"]]=numpy.log(prepared_encoded_data[:, feature_indices["log_total_charge"]]+1.0)
    
    # time starts at charged weighted mean time, and not at trigger start time
    median_times=numpy.array(median_times)-charge_weighted_time_mean

    #print("overall charege ", overall_charge_in_event)

    return prepared_encoded_data, mean_cog_pos, string_id_list, om_id_list, module_type_list, overall_charge_in_event, median_times
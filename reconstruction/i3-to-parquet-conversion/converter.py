import collections
import yaml
import tqdm
import os
import numpy
import pandas
# @TODO: check this import
from importlib.metadata import version
if(version("awkward")[0]=="1"):
    import awkward._v2 as awkward
else:
    import awkward

from icecube import dataio, dataclasses
from icecube.ml_suite import EventFeatureFactory

import i3_methods
import feature_configs

class Converter(object):
    def __init__(self,
                filenames,
                target_folder,
                encoding_type="mod_harnisch",
                pulse_series_name="InIcePulses",
                sub_event_stream=None,
                gcdfile="/data/sim/sim-new/downloads/GCD/GeoCalibDetectorStatus_2021.Run135903.T00S1.Pass2_V1b_Snow211115.i3.gz",
                num_events_per_file=1000, # how many events per .pq file
                num_per_row_group=100, # 100 seems a reasonable size
                is_llp=False, # is the dataset an LLP simulation?
                ):
        # input and output filenames
        self.filenames = filenames
        self.num_files = len(self.filenames)
        self.target_filenames=[] # output .pq filenames (different sizes than input files)
        self.target_folder=target_folder # where to save .pq files

        # conversion settings
        self.encoding_type = encoding_type
        self.pulse_series_name = pulse_series_name
        self.sub_event_stream = sub_event_stream
        assert self.sub_event_stream in ["InIceSplit", "NullSplit", None], "Unknown split " + self.sub_event_stream
        self.num_events_per_file=num_events_per_file
        self.num_per_row_group=num_per_row_group
        assert(self.num_events_per_file%self.num_per_row_group==0), (self.num_events_per_file, self.num_per_row_group)
        self.num_rowgroups=self.num_events_per_file/self.num_per_row_group
        self.is_llp = is_llp
        
        
        # index dictionary to hold meta info about .pq files
        self.total_index_info=dict()
        self.total_index_info["file_index"]=[]
        self.total_index_info["index_within_file"]=[]
        self.total_index_info["rowgroup_index_within_file"]=[]
        self.total_index_info["event_id"]=[]
        self.total_index_info["run_id"]=[]
        self.total_index_info["muon_energy"]=[]
        self.total_index_info["muon_zenith"]=[]
        self.total_index_info["muon_length"]=[]

        # MC weighting info
        self.total_weight_info=dict()
        self.total_weight_info["I3MCWeightDict"]=dict()

        # extract geometry and DOM efficienies (rde = relative dom efficieency)
        self.i3_geo, self.geo, self.rde_dict = i3_methods.extract_geometry_n_rdes(gcdfile)
        
        # add geometry surface
        self.surface = i3_methods.MakeSurface(gcdfile, 0.)
        #####################################################

        print("####################################")
        print(" Starting Conversion ..... ")
        print(" encoding type: ", encoding_type)
        print(" pulses: ", self.pulse_series_name)
        print(" gcdfile: ", gcdfile)
        print(" events per file: ", self.num_events_per_file)
        print(" target folder: ", self.target_folder)
        print("####################################")
        
        # start conversion from i3 to parquet
        self.convert_all()

    def convert_all(self):
        # initiliaze buffer to hold encoded data and labels
        self.initialize_buffer()

        # create feature extractor based on encoding type
        self.feature_extractor, self.feature_indices = self.create_feature_objects()

        # buffered reading and conversion i3->pq
        total_num_written, \
        total_num_events_overall, \
        total_num_files_written = self.convert_i3_files()

        # create auxiliary files needed for re-weighting
        self.write_aux_files(total_num_written,
                            total_num_events_overall,
                            total_num_files_written)    

    def convert_i3_files(self):
        """ Main conversion loop.
        
        Iterate over all i3 files and fill the buffer.
        Once buffer is filled, write to .pq file.
        Buffer might be filled in the middle of an i3 file,
        but next iteration will continue from the last written event.
        
        """
        # set flags and counters
        total_num_written=0           # how many events written to pq?
        self.num_appended=0           # how many events in current buffer?
        self.cur_write_file_index=0   # which .pq file are we on?
        self.cur_file_f_counter=0     # which frame in current i3 file?
        self.cur_file_index=0         # which .i3 file are we on?
        self.is_first_frame_mc = None # is the first frame MC?

        # create progress bar for iterating through files
        pbar = tqdm.tqdm(total=self.num_files)

        ####### MAIN LOOP #######
        # iterate i3 files
        # @TODO: make this a for loop over self.filenames
        while(self.cur_file_index<self.num_files):
            # new .pq file so clear buffer
            self.clear_buffer()
            self.num_appended=0

            # fill buffer
            while( self.num_appended<self.num_events_per_file ) :
                if(self.cur_file_index>=self.num_files):
                    break

                print("opening file ... ", self.filenames[self.cur_file_index])
                f=dataio.I3File(self.filenames[self.cur_file_index])
                this_file_f_counter = 0

                # go through frames in i3 file
                while( self.num_appended<self.num_events_per_file and f.more() ):
                    # @TODO: option for P vs Q frame
                    # pop Q or P frame
                    try:
                        if self.sub_event_stream is not None:
                            frame = f.pop_physics()
                            # reach desired sub_event_stream
                            while(frame["I3EventHeader"].sub_event_stream != self.sub_event_stream):
                                frame = f.pop_physics()
                        else:
                            frame = f.pop_daq()
                    except:
                        print("could not pop frame")
                        break

                    # check that frame is good
                    if(not self.check_good_frame(frame)):
                        #print("Bad frame, skip:", frame)
                        continue

                    this_file_f_counter+=1
                    # if previous buffer was full before end of i3 file, then start from that point
                    if(this_file_f_counter>self.cur_file_f_counter):
                        self.cur_file_f_counter+=1
                        # fill buffer with event
                        self.add_event_to_buffer(frame)
                        self.num_appended += 1

                f.close()
                print("... closed file .. num appended %d - target max buffer %d" % \
                      (self.num_appended, self.num_events_per_file))
                
                # did we fill the buffer before the i3file was empty?
                if(self.num_appended==self.num_events_per_file):
                    print("reached max buffer ..")
                    print("CUR FILE FRAME COUNTER ", self.cur_file_f_counter)
                    # break out and fill new buffer from the next frame in this i3 file
                    break
                else:
                    # go to next i3 file and reset frame counter
                    self.cur_file_index+=1
                    self.cur_file_f_counter=0

                pbar.update(1)

            # is buffer full?
            if(self.num_appended==self.num_events_per_file):
                # save buffer to file
                self.save_buffer_to_file()
                self.cur_write_file_index+=1
                total_num_written+=self.num_appended
        #################################################

        pbar.close()

        # if excess events, trim the index dictionaries
        self.trim_dictionaries()
        
        print("tot num written", total_num_written)
        print("tot num + appended ", total_num_written+self.num_appended)
        print("cur write file index", self.cur_write_file_index)

        return total_num_written, total_num_written+self.num_appended, self.cur_write_file_index

    def add_event_to_buffer(self, frame):
        # extract pulse data, etc., from frame
        event_info = i3_methods.obtain_encoded_data(frame, 
                                                    self.pulse_series_name,
                                                    self.geo, 
                                                    self.rde_dict,
                                                    feature_extractor=self.feature_extractor,
                                                    feature_indices=self.feature_indices
                                                    )
        # unpack return tuple
        final_data, mean_cog, string_list, om_list, module_type_list, overall_charge, median_times = event_info

        # event info
        self.buffer["data_encoded"].append(final_data)
        self.buffer["data_weighted_medians"].append(median_times)
        self.buffer["data_cog_mean"].append(mean_cog)
        self.buffer["string_list"].append(string_list)
        self.buffer["om_list"].append(om_list)
        self.buffer["module_type_list"].append(module_type_list)
        self.buffer["totcharge"].append(overall_charge)
        self.buffer["nch"].append(len(string_list))
        self.buffer["run_id"].append(frame["I3EventHeader"].run_id)
        self.buffer["event_id"].append(frame["I3EventHeader"].event_id)

        ## index information
        self.total_index_info["index_within_file"].append(self.num_appended)
        self.total_index_info["rowgroup_index_within_file"].append(int(self.num_appended//self.num_per_row_group))
        self.total_index_info["file_index"].append(self.cur_write_file_index)
        self.total_index_info["run_id"].append(frame["I3EventHeader"].run_id)
        self.total_index_info["event_id"].append(frame["I3EventHeader"].event_id)
        
        if(self.is_frame_mc(frame)):
            # add extra MC info
            self.add_MC_info(frame)
        if(self.is_llp):
            self.add_llp_info_to_buffer(frame)
            
    def add_llp_info_to_buffer(self, frame):
        # get data from frame["LLPInfo"]
        llp_data = i3_methods.obtain_llp_data(frame)
        # unpack
        production, decay, direction, \
        gap_length, fractional_energy, llp_energy, \
        decay_asymmetry = llp_data
        # add to buffer
        self.buffer["llp_prod_x"].append(production.x)
        self.buffer["llp_prod_y"].append(production.y)
        self.buffer["llp_prod_z"].append(production.z)
        self.buffer["llp_decay_x"].append(decay.x)
        self.buffer["llp_decay_y"].append(decay.y)
        self.buffer["llp_decay_z"].append(decay.z)

        self.buffer["llp_gap_length"].append(gap_length)
        self.buffer["llp_fractional_energy"].append(fractional_energy)
        self.buffer["llp_energy"].append(llp_energy)
        self.buffer["llp_decay_asymmetry"].append(decay_asymmetry) 
    
    def add_MC_info(self, frame):
        """ Add muon spectrum info to buffer. """
        mcinfo = i3_methods.obtain_MC_info(frame, self.surface)
        self.total_index_info["muon_energy"].append(mcinfo[0])
        self.total_index_info["muon_zenith"].append(mcinfo[1])
        self.total_index_info["muon_length"].append(mcinfo[2])
    
    def write_aux_files(self, total_num_written, total_num_events_overall, total_num_files_written):
        """ Write files used for re-weighting.

        Since we only save a multiple of num_events_per_file
        the weighting will be off. The following files are used to
        account for the events that were left out.

        """
        ############ the next 3 weighting files are needed for correct re-weighting

        print("Total num written, total events overall, total files written:")
        print(total_num_written, total_num_events_overall, total_num_files_written)
        ## also write weightning information into pandas file (to be used with weightsim)
        df=pandas.DataFrame(data=self.total_weight_info)    
        df.to_parquet(path=os.path.join(self.target_folder, "weightfile.pq"))
        
        ## write index file which holds index information
        df=pandas.DataFrame(data=self.total_index_info)    
        df.to_parquet(path=os.path.join(self.target_folder, "indexfile.pq"))
            
        effective_no_i3_per_pq=float(len(self.filenames))*float(total_num_written)/float(total_num_events_overall)
        effective_no_i3_per_pq/=float(total_num_files_written)

        ## write index_to_string identifier file

        effective_events_per_file=self.num_events_per_file
        if(effective_events_per_file==-1):
            effective_events_per_file=total_num_written

        df=pandas.DataFrame(data={"filename": self.target_filenames,
                                  "effective_num_i3_files_per_pq": effective_no_i3_per_pq,
                                  "num_events_per_file": effective_events_per_file})
        df.to_parquet(path=os.path.join(self.target_folder, "filelist.pq"))
        df.to_csv(os.path.join(self.target_folder, "filelist.csv"))

        ### write feature order to config yaml
        with open(os.path.join(self.target_folder, "feature_indices.yaml"), 'w') as file:
            yaml.dump(self.feature_indices, file)

        ############
        print("Total num events written to files .. ", total_num_written)

    def create_feature_objects(self):
        """ Create feature extractor from icecube.ml_suite.EventFeatureFactory
            and feature indices dictionary.

            Used to extract pulse data etc. from frame.
            feature_indices is used to label the returned feature
            vector from feature_extractor
        """
        # features
        feature_extractor=None
        feature_indices=dict()

        # LOAD FEATURES CONFIGURATION
        ft_config=yaml.safe_load(feature_configs.feature_configs[self.encoding_type].replace("PULSEKEY_PLACEHOLDER", self.pulse_series_name))
        print("FT CONFIG", ft_config)

        # indices for DOM position xyz, since we are manually adding position
        feature_indices["position"]=list(range(0,3))
        last_offset = 3 # account for xyz in the prepended feature vector

        # iterate all features and fix indices
        for item in ft_config["feature_config"]["features"]:
            
            # charge stuff
            # TotalCharge -> log(TotalCharge + 1)
            if(item["class"]=="TotalCharge"):
                if("log_charges" not in feature_indices.keys()):
                    feature_indices["log_charges"]=[]

                feature_indices["log_charges"].extend(list(range(last_offset, last_offset+1)))
                feature_indices["log_total_charge"]=last_offset
                last_offset+=1

            elif(item["class"]=="ChargeUntilT"):
                if("log_charges" not in feature_indices.keys()):
                    feature_indices["log_charges"]=[]

                num_times=len(item["kwargs"]["times"])

                feature_indices["log_charges"].extend(list(range(last_offset, last_offset+num_times)))
                last_offset+=num_times

            ## abs times
            elif(item["class"]=="TFirstPulse"):
                if("abs_times" not in feature_indices.keys()):
                    feature_indices["abs_times"]=[]


                feature_indices["abs_times"].extend(list(range(last_offset, last_offset+1)))
                last_offset+=1

            elif(item["class"]=="TLastPulse"):
                if("abs_times" not in feature_indices.keys()):
                    feature_indices["abs_times"]=[]


                feature_indices["abs_times"].extend(list(range(last_offset, last_offset+1)))
                last_offset+=1

            elif(item["class"]=="TimeAtChargePercentile"):
                if("abs_times" not in feature_indices.keys()):
                    feature_indices["abs_times"]=[]


                num_times=len(item["kwargs"]["percentiles"])

                feature_indices["abs_times"].extend(list(range(last_offset, last_offset+num_times)))
                last_offset+=num_times

            elif(item["class"]=="TSpread"):

                if("time_widths" not in feature_indices.keys()):
                    feature_indices["time_widths"]=[]

                feature_indices["time_widths"].extend(list(range(last_offset, last_offset+1)))
                last_offset+=1
            elif(item["class"]=="ChargeWeightedMean"):

                if("abs_times" not in feature_indices.keys()):
                    feature_indices["abs_times"]=[]

                feature_indices["abs_times"].extend(list(range(last_offset, last_offset+1)))
                last_offset+=1
            elif(item["class"]=="ChargeWeightedStd"):

                if("time_widths" not in feature_indices.keys()):
                    feature_indices["time_widths"]=[]

                feature_indices["time_widths"].extend(list(range(last_offset, last_offset+1)))
                last_offset+=1
            
        feature_extractor = EventFeatureFactory(ft_config).make_feature_extractor()
        return feature_extractor, feature_indices

    def trim_dictionaries(self):
        """ Match size of index info dictionaries to file size.
        
        Only multiples of num_event_per_file are saved, so trim off
        excess events from the index dictionaries.

        This is only applicable if total number of events is not
        a multiple of num_events_per_file.
        
        """
        # subtract from the total weight lists the currently num_appended to be at same length as full dict
        if(self.num_appended>0):
            if(self.is_first_frame_mc):
                for k in self.total_weight_info["I3MCWeightDict"].keys():
                    self.total_weight_info["I3MCWeightDict"][k]=numpy.array(self.total_weight_info["I3MCWeightDict"][k][:-self.num_appended])
            for key, value in self.total_index_info.items():
                self.total_index_info[key]=self.total_index_info[key][:-self.num_appended]

    def check_good_frame(self, frame):
        # @TODO: implement
        
        # check that pulse series is in frame
        if(not self.pulse_series_name in frame):
            print("No", self.pulse_series_name, "in frame with subeventstream", frame["I3EventHeader"].sub_event_stream)
            return False
        
        # is it MC frame?
        is_mc = self.is_frame_mc(frame)

        if(self.is_first_frame_mc is None):
            self.is_first_frame_mc=is_mc

        ## check that all files are similar (MC or data, but not both)
        assert(self.is_first_frame_mc==is_mc)
        # check that you didn't lie when you said dataset has LLP
        assert(self.is_llp == self.is_frame_llp(frame))

        return True
    
    def is_frame_mc(self, frame):
        # @TODO: is this a good way to check MC?
        return frame.Has("MMCTrackList")
    
    def is_frame_llp(self,frame):
        return frame.Has("LLPInfo")
    
    def initialize_buffer(self):
        """ Create empty buffer that holds event data."""
        #@TODO: what to put in here?
        #@TODO: add LLP info
        self.buffer = collections.OrderedDict()
        self.buffer["data_encoded"]=[] # this is input to network (but scale times etc first)
        self.buffer["data_weighted_medians"]=[]
        self.buffer["data_cog_mean"]=[]

        self.buffer["string_list"]=[]
        self.buffer["om_list"]=[]
        self.buffer["module_type_list"]=[]

        self.buffer["totcharge"]=[]
        self.buffer["nch"]=[]

        self.buffer["run_id"]=[]
        self.buffer["event_id"]=[]

        # LLP info
        if self.is_llp:
            self.buffer["llp_prod_x"] = []
            self.buffer["llp_prod_y"] = []
            self.buffer["llp_prod_z"] = []
    
            self.buffer["llp_decay_x"] = []
            self.buffer["llp_decay_y"] = []
            self.buffer["llp_decay_z"] = []

            self.buffer["llp_gap_length"] = []
            self.buffer["llp_fractional_energy"] = []
            self.buffer["llp_energy"] = []
            self.buffer["llp_decay_asymmetry"] = []
 
    def clear_buffer(self):
        for k in self.buffer.keys():
            self.buffer[k].clear()

    def save_buffer_to_file(self):
        # buffer dictionary -> awkward array
        full_arr=awkward.Array(self.buffer)

        # create filename
        splits=self.filenames[self.cur_file_index].split(".")
        ## remove gz and i3 ending etc
        if(splits[-1]=="i3" or splits[-1]=="gz" or splits[-1]=="bz2" or splits[-1]=="zst"):
            splits=splits[:-1]
        if(splits[-1]=="i3" or splits[-1]=="gz" or splits[-1]=="bz2" or splits[-1]=="zst"):
            splits=splits[:-1]
        # put filename back together, now without file extensions
        raw_filename=os.path.basename(".".join(splits))+(".%.6d" % self.cur_write_file_index)
        target_filename=os.path.join(self.target_folder, raw_filename) + ".pq"

        # save to parquet
        print("Writing buffer to", target_filename)
        awkward.to_parquet(full_arr, target_filename, row_group_size=self.num_per_row_group)
        self.target_filenames.append(target_filename)

        return target_filename

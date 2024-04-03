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

from icecube import dataio
from icecube.ml_suite import EventFeatureFactory

import i3_methods # @TODO: change name
import feature_configs # @TODO: change name


class Converter(object):
    def __init__(self,
                filenames,
                target_folder,
                encoding_type="mod_harnisch",
                pulse_series_name="InIcePulses",
                gcdfile="/data/sim/sim-new/downloads/GCD/GeoCalibDetectorStatus_2021.Run135903.T00S1.Pass2_V1b_Snow211115.i3.gz",
                num_events_per_file=1000):
        # create filenames
        self.filenames = filenames
        self.num_files = len(self.filenames)
        # will hold target .pq filenames (of potentially different sizes than input files)
        self.target_filenames=[]
        self.target_folder=target_folder

        # create gcd surface
        padding = 0.
        #self._surface = axel_i3_methods.MakeSurface(gcdfile, padding)
        
        # dictionaries
        self.total_index_info=dict()
        self.total_index_info["file_index"]=[]
        self.total_index_info["index_within_file"]=[]
        self.total_index_info["rowgroup_index_within_file"]=[]
        self.total_index_info["event_id"]=[]
        self.total_index_info["run_id"]=[]
        # self.total_index_info["log10_deposited_energy"]=[]
        
        self.total_weight_info=dict()
        self.total_weight_info["I3MCWeightDict"]=dict()
        # initialize conversion variables
        self.encoding_type = encoding_type
        self.pulse_series_name = pulse_series_name
        self.num_events_per_file=num_events_per_file
        # 100 seems a reasonable size
        self.num_per_row_group=50 #@TODO: fix this 
        assert(self.num_events_per_file%self.num_per_row_group==0), (self.num_events_per_file, self.num_per_row_group)
        self.num_rowgroups=self.num_events_per_file/self.num_per_row_group

        # extract geometry and DOM efficienies (rde = relative dom efficieency)
        self.i3_geo, self.geo, self.rde_dict = i3_methods.extract_geometry_n_rdes(gcdfile)
        #####################################################

        print("####################################")
        print(" Starting Conversion ..... ")
        #print(" dir: %s" % file_directory)
        print(" encoding type: ", encoding_type)
        print(" pulses: ", self.pulse_series_name)
        #print(" filter: ", filter_name)
        #print(" subeventstream: ", subeventstream)
        print("####################################")
        
        # start conversion from i3 to parquet
        self.convert_all()

    def convert_all(self):
        # initiliaze buffer to hold encoded data and labels
        self._initialize_buffer()

        # buffered reading and conversion i3->pq
        total_num_written, \
        total_num_events_overall, \
        total_num_files_written = self.extract_into_similar_sized_files()

        # create files needed for re-weighting
        self.write_aux_files(total_num_written, total_num_events_overall, total_num_files_written)        

    def write_aux_files(self, total_num_written, total_num_events_overall, total_num_files_written):
        ############ the next 3 weighting files are needed for correct re-weighting

        print("Total num written, total events overall, total files written:")
        print(total_num_written, total_num_events_overall, total_num_files_written)
        ## also write weightning information into pandas file (to be used with weightsim)
        for k in self.total_weight_info.keys():
            print(k, len(self.total_weight_info[k]))
        df=pandas.DataFrame(data=self.total_weight_info)    
        df.to_parquet(path=os.path.join(self.target_folder, "weightfile.pq"))
        
        ## write index file which holds index information
        for k in self.total_index_info.keys():
            print(k, len(self.total_index_info[k]))
        df=pandas.DataFrame(data=self.total_index_info)    
        df.to_parquet(path=os.path.join(self.target_folder, "indexfile.pq"))
            
        effective_no_i3_per_pq=float(len(self.filenames))*float(total_num_written)/float(total_num_events_overall)

        effective_no_i3_per_pq/=float(total_num_files_written)

        ## write index_to_string identifier file

        effective_events_per_file=self.num_events_per_file
        if(effective_events_per_file==-1):
            effective_events_per_file=total_num_written

        df=pandas.DataFrame(data={"filename": self.target_filenames, "effective_num_i3_files_per_pq": effective_no_i3_per_pq, "num_events_per_file": effective_events_per_file})
        df.to_parquet(path=os.path.join(self.target_folder, "filelist.pq"))
        df.to_csv(os.path.join(self.target_folder, "filelist.csv"))

        ### write feature order to config yaml
        with open(os.path.join(self.target_folder, "feature_indices.yaml"), 'w') as file:
            yaml.dump(self.feature_indices, file)

        ############
        print("Total num events written to files .. ", total_num_written)

    def extract_into_similar_sized_files(self):
        # create feature objects based on encoding type
        self.feature_extractor, self.feature_indices = self.create_feature_objects()

        # main loop
        return self.go_through_i3_files()

    def go_through_i3_files(self):
        # set flags and counters
        total_num_written=0
        self.num_appended=0
        self.cur_write_file_index=0
        self.cur_file_p_counter=0
        self.cur_file_index=0
        self.is_first_frame_mc = None
        # create progress bar for iterating through files
        pbar = tqdm.tqdm(total=self.num_files)

        ### MAIN LOOP ###
        # iterate i3 files
        # @TODO: make this a for loop over self.filenames
        while(self.cur_file_index<self.num_files):
            # new file so clear buffer
            self.clear_buffer()
            self.num_appended=0

            # fill buffer
            while( self.num_appended<self.num_events_per_file ) :
                
                if(self.cur_file_index>=self.num_files):
                    break
                
                print("opening file ... ", self.filenames[self.cur_file_index])
                f=dataio.I3File(self.filenames[self.cur_file_index])
                this_file_p_counter = 0

                # go through frames in i3 file
                while( self.num_appended<self.num_events_per_file and f.more() ):
                    # get frame
                    # @TODO: option for P vs Q frame
                    try:
                        frame = f.pop_daq()
                    except:
                        print("could not pop daq frame")
                        break

                    # check that frame is good
                    if(not self.check_good_frame(frame)):
                        print("Bad frame, skip:", frame)
                        continue

                    this_file_p_counter+=1
                    # if previous buffer was full in the middle of this i3 file, then start from that point now
                    if(this_file_p_counter>self.cur_file_p_counter):
                        ## new p frame - extract info
                        self.cur_file_p_counter+=1

                        # fill buffer with event
                        self.add_event_to_buffer(frame)
                        self.num_appended += 1

                f.close()
                print("... closed file .. num appended %d - target max buffer %d" % \
                      (self.num_appended, self.num_events_per_file))
                
                # did we fill the buffer before the i3file was empty?
                if(self.num_appended==self.num_events_per_file):
                    print("reached max buffer ..")
                    print("CUR FILE P COUNTER ", self.cur_file_p_counter)
                    # break out again if we have filled the buffer
                    break
                else:
                    # go to next i3 file and reset frame counter
                    self.cur_file_index+=1
                    self.cur_file_p_counter=0

                pbar.update(1)

            ## go back with deque index by as many items as have been appended
            #self.cur_deque_index-=num_appended
            #print("num ", num_appended, "evpf", self.num_events_per_file)
            if(self.num_appended==self.num_events_per_file):
                ## we hit exactly the desired number ... write file 

                splits=self.filenames[self.cur_file_index].split(".")
  
                ## remove gz and i3 ending etc
                if(splits[-1]=="i3" or splits[-1]=="gz" or splits[-1]=="bz2" or splits[-1]=="zst"):
                    splits=splits[:-1]
                if(splits[-1]=="i3" or splits[-1]=="gz" or splits[-1]=="bz2" or splits[-1]=="zst"):
                    splits=splits[:-1]
                ## take all splits
                raw_filename=os.path.basename(".".join(splits))+(".%.6d" % self.cur_write_file_index)
                new_target_filename=os.path.join(self.target_folder, raw_filename)

                # SAVE BUFFER TO FILE
                print(new_target_filename)
                final_filename=self._save_buffer_to_file( new_target_filename)
                
                self.target_filenames.append(final_filename)

                ## sql has only a single files for example
                #if(self.write_multiple_files):
                self.cur_write_file_index+=1

                total_num_written+=self.num_appended
                #print("cur write file index", self.cur_write_file_index)

        pbar.close()

         ## subtract from the total weight lists the currently num_appended to be at same length as full dict

        if(self.num_appended>0):

            if(self.is_first_frame_mc):
                for k in self.total_weight_info["I3MCWeightDict"].keys():
                    self.total_weight_info["I3MCWeightDict"][k]=numpy.array(self.total_weight_info["I3MCWeightDict"][k][:-self.num_appended])

            self.total_index_info["index_within_file"]=self.total_index_info["index_within_file"][:-self.num_appended]
            self.total_index_info["rowgroup_index_within_file"]=self.total_index_info["rowgroup_index_within_file"][:-self.num_appended]
            self.total_index_info["file_index"]=self.total_index_info["file_index"][:-self.num_appended]

            ## also add run_id / event_id for crosscheck .. dont need it really

            self.total_index_info["run_id"]=self.total_index_info["run_id"][:-self.num_appended]
            self.total_index_info["event_id"]=self.total_index_info["event_id"][:-self.num_appended]
            # self.total_index_info["log10_deposited_energy"]=self.total_index_info["log10_deposited_energy"][:-self.num_appended]
        print("tot num written", total_num_written)
        print("tot num + appended ", total_num_written+self.num_appended)
        print("cur write file index", self.cur_write_file_index)

        return total_num_written, total_num_written+self.num_appended, self.cur_write_file_index

    def clear_buffer(self):
        for k in self.buffer.keys():
            self.buffer[k].clear()

    def add_event_to_buffer(self, frame):
        
        # get pulse data
        # event_info = final_data, mean_cog, string_list, om_list, module_type_list, overall_charge, median_times
        event_info = i3_methods.obtain_encoded_data(frame, 
                                                    #data_overview, final_data, weighted_time_mean, weighted_time_median, mean_cog, median_cog, string_list, om_list, overall_charge=i3_methods.obtain_encoded_data(phys_frame, 
                                                    self.pulse_series_name, # afterpulse cleaned map if cleaning is used, otherwise this will be the normal map
                                                    self.geo, 
                                                    self.rde_dict,
                                                    feature_extractor=self.feature_extractor,
                                                    feature_indices=self.feature_indices
                                                    )
        # unpack return tuple
        final_data, mean_cog, string_list, om_list, module_type_list, overall_charge, median_times = event_info

        self.buffer["data_encoded"].append(final_data)
        self.buffer["data_weighted_medians"].append(median_times)
        
        self.buffer["data_cog_mean"].append(mean_cog)

        self.buffer["string_list"].append(string_list)
        self.buffer["om_list"].append(om_list)
        self.buffer["module_type_list"].append(module_type_list)

        self.buffer["totcharge"].append(overall_charge)
        self.buffer["nch"].append(len(string_list))

        ## also get event id/runid

        self.buffer["run_id"].append(frame["I3EventHeader"].run_id)
        self.buffer["event_id"].append(frame["I3EventHeader"].event_id)

        ## index information
        self.total_index_info["index_within_file"].append(self.num_appended)
        self.total_index_info["rowgroup_index_within_file"].append(int(self.num_appended//self.num_per_row_group))
        self.total_index_info["file_index"].append(self.cur_write_file_index)

        ## also add run_id / event_id for crosscheck .. dont need it really

        self.total_index_info["run_id"].append(frame["I3EventHeader"].run_id)
        self.total_index_info["event_id"].append(frame["I3EventHeader"].event_id)
        
        if(self.is_frame_mc(frame)):
            # add extra MC info
            # @TODO: implement
            pass
        if(self.is_frame_llp(frame)):
            # add extra MC info
            # @TODO: implement
            pass

    def check_good_frame(self, frame):
        # @TODO: implement

        # is it MC frame?
        is_mc = self.is_frame_mc(frame)

        if(self.is_first_frame_mc is None):
            self.is_first_frame_mc=is_mc

        ## check that all files are similar (MC or data, but not both)
        assert(self.is_first_frame_mc==is_mc)

        return True
    
    def is_frame_mc(self, frame):
        # @TODO: is this a good way to check MC?
        return frame.Has("MMCTrackList")
    
    def is_frame_llp(self,frame):
        return frame.Has("LLPInfo")
    
    def create_feature_objects(self):
        """ Create feature extractor and feature indices
        """
        # features
        feature_extractor=None
        feature_indices=dict()

        # LOAD FEATURES CONFIGURATION
        ft_config=yaml.safe_load(feature_configs.feature_configs[self.encoding_type].replace("PULSEKEY_PLACEHOLDER", self.pulse_series_name))
        print("FT CONFIG", ft_config)
        # indices for DOM position xyz
        feature_indices["position"]=list(range(0,3))
        
        # this is for adding position to the feature vector from feature_extractor
        last_offset = 3 

        # iterate all features and fix indices
        for item in ft_config["feature_config"]["features"]:
            
            ## charge stuff
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
    
    def _initialize_buffer(self):
        """ Create empty buffer that holds event data."""
        #@TODO: what to put in here?
        #@TODO: add LLP info
        self.buffer = collections.OrderedDict()
        self.buffer["data_encoded"]=[] # this is input to network (but scale etc first)
        self.buffer["data_weighted_medians"]=[]
        self.buffer["data_cog_mean"]=[]

        self.buffer["string_list"]=[]
        self.buffer["om_list"]=[]
        self.buffer["module_type_list"]=[]

        # self.buffer["length"]=[]

        self.buffer["totcharge"]=[]
        self.buffer["nch"]=[]

        # self.buffer["interaction_type"]=[]
        self.buffer["run_id"]=[]
        self.buffer["event_id"]=[]

        # LLP info
        # self.buffer["llp_prod_x"] = []
        # self.buffer["llp_prod_y"] = []
        # self.buffer["llp_prod_z"] = []

        # self.buffer["llp_decay_x"] = []
        # self.buffer["llp_decay_y"] = []
        # self.buffer["llp_decay_z"] = []


    def _save_buffer_to_file(self, base_filename):

        # print("self buffer", self.buffer)
        full_arr=awkward.Array(self.buffer)

        awkward.to_parquet(full_arr, base_filename+".pq", row_group_size=self.num_per_row_group)

        return base_filename+".pq"


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Define a class to import, preprocess and epoch the data saved by eyelink.

@author: Ceated by Florent Meyniel / adapted by Maëva L'Hôtellier & Aziza Chebil / Enhanced by Yannaël Bossard
"""

import re
import pandas as pd
import numpy as np
import os
import copy
from scipy import signal, stats
import matplotlib.pyplot as plt
import warnings
import scipy
from scipy.interpolate import interp1d
import math #Useful if you want to check NaN-value
import time #Useful for debugging



class pupil:
    def __init__(self, filename=None, directory=None):
        self.filename = filename
        self.directory = directory

    def import_data(self, filename=None, events_warner=None, directory=None, eye='left',
                    col=[0, 1, 2],
                    col_name=['x_pos', 'y_pos', 'diameter'],
                    eyetracker='tobii'):
        """ Import the data.
            - filename: name of file in the directory (supersedes the one in initialization)
            - events_warner: string which permits to track lines of text in the input file.
            - directory: full path to the directory containing the file (supersedes the one in 
              initialization)
            - eye: Pick the eye you want to study. Only data from this eye will be return. 
              There is a function (pick_best_eye) to help you to choose the best eye
            - col: column to extract from the file (after the time stamp)
            - names of the corresponding columns
            - eyetracker : define the regexp according to the eyetracker specified: regular expression
              to parse the columns of the file. 1st element is the time_stamp, following elements are 
              data points (1st data point correspond to col[0] and col_name[0])
        It returns:
            - the events collected by eyelink
            - the blinks (onsets and offsets) detected by eyelink.
                we consider a blink as a NaN in diameter of picked eye
            - the time_stamp (ms) of the data (each data point)
            - time_stamp (s) relative to first data point
            - the data (as a panda dataframe)
            - nb_diam_nan_left : this is the number of nan in diameter column for the left eye. this is used for pick_best_eye
            - nb_diam_nan_right : this is the number of nan in diameter column for the right eye. this is used for pick_best
            - list_diameter_nan: this is the boolean list with True if the diameter of the picked eye is NaN. It 
                will be usefull for counting the number of NaN per epoch
            - nan_ratio: this is the ratio of nan in diameter column for the eye.
            - nb_diameter_nan: this is the number of nan in diameter column for the picked eye.

        The datafile should be an ascii file (best) or an edf file. In the latter case, the script
        will attempt to convert it to ascii (on linux).
        """

        #IMPORTANT blinks recognition are only based on the NaN for pupil diameter (right or left according to what you chose)
        # If youre not studing the diameter but the position, you should change a bit the blink function and the counter of NaN

        if eyetracker == 'tobii':
            regexp = '(\d+) \t \((-?)\d*.\d*,\s(-?)\d*.\d*\) \t \((-?)\d*.\d*,' + \
                           '\s(-?)\d*.\d*\) \t (\d*.\d*) \t (\d*.\d*)\t \n'
            self.time_scale = 1000000

        elif eyetracker == 'eyelink':
            regexp = "(\d+)(\t\s*(-?)\d*.\d*)(\t\s*(-?)\d*.\d*)(\t\s*\d+.\d+)(\t...\n)"
            self.time_scale = 1000

        else:
            raise ValueError('Incorrect eyetracker specification. Did you mean "tobii" or "eyelink" ?')

        # Locate file
        if filename is None:
            if self.filename is None:
                raise ValueError('no filename provided: cannot import data')
        else:
            self.filename = filename
        if directory is None:
            # will look for the file in the current folder
            directory = os.getcwd()

        filename = os.path.join(directory, filename)
        if not os.path.isfile(filename):
            raise ValueError('Could not find ', filename)

        # Read file and get values
        events, time_stamp = [], []
        blinks_counter = [0]
        blink_start, blink_end = [], []
        data = [[], [], []]
        
        # Counting the number of NaN in the used column
        list_nan_left_diameter = []
        list_nan_right_diameter = []

        if eyetracker=='eyelink':
            # If not an ascii file:
            if filename.split('.')[-1] == 'edf':
                if os.path.isfile(filename[:-3]+'asc'):
                    # Replace by the same ascii file, it exist
                    filename = filename[:-3]+'asc'
                else:
                    # try to convert it
                    os.system('cd eyetracker_tools;'+f"./edf2asc {filename}")
                    filename = filename[:-3]+'asc'
                    if not os.path.isfile(filename):
                        raise ValueError('I tried to convert your edf file but I failed')

            # Read file and get values
            events, time_stamp = [], []
            blink_start, blink_end = [], []
            data = [[], [], []]
            with open(filename, "r") as file:
                line = file.readline()
                while line:
                    if line[0:3] == 'MSG':
                        # get the message that was sent to the eyetracker
                        info = line.split('\t')[1].strip()
                        event_time = int(info.split(' ')[0])
                        event_msg = " ".join(info.split(' ')[1:])
                        events.append({'time': event_time, 'msg': event_msg})

                    elif re.match(regexp, line) is not None:
                        # get the value that was saved by the eyetracker
                        info = line.rstrip('\t...\n').split('\t')
                        time_stamp.append(int(info.pop(0)))
                        for k, val in enumerate(info):
                            try:
                                data[k].append(float(val))
                            except:
                                data[k].append(float(0))

                    elif line[0:6] == 'SBLINK':
                        blink_start.append(int(line[9:].strip()))
                    elif line[0:6] == 'EBLINK':
                        blink_end.append(int(line[9:].strip().split('\t')[1]))
                    line = file.readline()
            self.events = events
            self.blink_start = np.array(blink_start)
            self.blink_end = np.array(blink_end)

            # Convert into a panda dataframe
            self.data = pd.DataFrame({'time_stamp': time_stamp,
                                        'time_stamp_rel_s': (np.array(time_stamp)
                                                            - time_stamp[0])/1000})
            col = [col] if type(col) is not list else col
            col_name = [col_name] if type(col_name) is not list else col_name
            for k, name in zip(col, col_name):
                self.data[name] = data[k]

            # get sampling rate (the time_stamp is in ms)
            self.recording_fs = 1000/(np.round(np.mean(np.diff(time_stamp))))

        else:
            with open(filename, "r", encoding="ISO-8859-1") as file:
                lines = file.readlines()
                for line in lines:
                    if events_warner in line:
                        info = line.split(' ')
                        event_time = info[0]
                        event_msg = info[1:]
                        events.append({'time': event_time, 'msg': event_msg})

                    if re.match(regexp, str(line)) is not None:
                        # get the values saved by the eyetracker
                        info = line.rstrip('\t...\n').split('\t')
                        time_stamp.append(int(info.pop(0)))

                        #Create boolean lists to calculate the rate of Nan per epoch (False for no Nan and True for NaN)
                        list_nan_left_diameter.append(False)
                        list_nan_right_diameter.append(False)

                        if eye == 'left':
                            del info[0], info[1]
                            
                        if eye == 'right':
                            del info[1], info[2]
                            
                        #  Clean info
                        info = info[0].split(',') + info[1: -1]
                        to_remove = "()"
                        pattern = '[' + to_remove + ']'
                        for k, val in enumerate(info):
                            val = re.sub(pattern, "", val)
                            try:
                                data[k].append(float(val))
                            except:
                                data[k].append('')

                        if blinks_counter[-1] == 1:
                            blink_end.append(time_stamp[-1])
                        blinks_counter.append(0)


                    # A NaN in line doesn't imply a blink, it can be only one eye which is not captured 
                    elif "nan" in line:
                        
                        info = line.rstrip('\t...\n').split('\t')
                        time_stamp.append(int(info[0]))

                        diameter_left = info[-2]
                        diameter_right = info[-3]

                        left_pos = info[-4]
                        left_pos = left_pos.strip()
                        left_pos = left_pos[1:-1]
                        left_pos = left_pos.split(',')
                        pos_x_left = left_pos[0]
                        pos_y_left = left_pos[1]

                        right_pos = info[-5]
                        right_pos = right_pos.strip()
                        right_pos = right_pos[1:-1]
                        right_pos = right_pos.split(',')
                        pos_x_right = right_pos[0]
                        pos_y_right = right_pos[1]
                        
                        
                        # Check if values are correct, not a whitespace neither a NaN
                        if not (isinstance(float(diameter_left), float) or math.isnan(float(diameter_left))):
                            print(" diameter_left = ", diameter_left)
                            warnings.warn("Attention, Il y a un problème dans regexp left_eye !")
                        if not (isinstance(float(diameter_right), float) or math.isnan(float(diameter_right))):
                            print( " diameter_right = ", diameter_right)
                            warnings.warn("Attention, Il y a un problème dans regexp right_eye !")
                        
                        # Add True to boolean list if there's a NaN diameter
                        if math.isnan(float(diameter_left)):
                            list_nan_left_diameter.append(True)
                            if eye == 'left':
                                if blinks_counter[-1] == 0:
                                    blink_start.append(int(info.pop(0)))
                                blinks_counter.append(1)
                        else:
                            list_nan_left_diameter.append(False)
                            if eye == 'left':
                                if blinks_counter[-1] == 1:
                                    blink_end.append(time_stamp[-1])
                                blinks_counter.append(0)

                        if math.isnan(float(diameter_right)):
                            list_nan_right_diameter.append(True)
                            if eye == 'right':
                                if blinks_counter[-1] == 0:
                                    blink_start.append(int(info.pop(0)))
                                blinks_counter.append(1)
                        else:
                            list_nan_right_diameter.append(False)
                            if eye == 'right':
                                if blinks_counter[-1] == 1:
                                    blink_end.append(time_stamp[-1])
                                blinks_counter.append(0)
    
                        info = [0, 0, 0]
                        for k, val in enumerate(info):
                            if k == 2: #index of the diameter in the column
                                if eye == 'left':
                                    if not math.isnan(float(diameter_left)):
                                        data[k].append(float(diameter_left))
                                    else:
                                        data[k].append(float(0))
                                elif eye == 'right':
                                    if not math.isnan(float(diameter_right)):
                                        data[k].append(float(diameter_right))
                                    else:
                                        data[k].append(float(0))
                            elif k == 0: #index of the Right position in the column
                                if eye == 'left':
                                    if not math.isnan(float(pos_x_left)):
                                        data[k].append(float(pos_x_left))
                                    else:
                                        data[k].append(float(0))
                                elif eye == 'right':
                                    if not math.isnan(float(pos_x_right)):
                                        data[k].append(float(pos_x_right))
                                    else:
                                        data[k].append(float(0))
                            elif k == 1: #index of the Left position in the column
                                if eye == 'left':
                                    if not math.isnan(float(pos_y_left)):
                                        data[k].append(float(pos_y_left))
                                    else:
                                        data[k].append(float(0))
                                elif eye == 'right':
                                    if not math.isnan(float(pos_y_right)):
                                        data[k].append(float(pos_y_right))
                                    else:
                                        data[k].append(float(0))


            
            #  Consider the last blink as ended if this is the end of the file
            if blinks_counter[-1] == 1:
                blink_end.append(time_stamp[-1])

            #  Store info
            self.events = events
            self.blink_start = np.array(blink_start)
            self.blink_end = np.array(blink_end)

            # Convert into a panda dataframe
            self.data = pd.DataFrame({'time_stamp': time_stamp,
                                    'time_stamp_rel_s': (np.array(time_stamp) - time_stamp[0])/self.time_scale})
            col = [col] if type(col) is not list else col

            col_name = [col_name] if type(col_name) is not list else col_name

            for k, name in zip(col, col_name):
                self.data[name] = data[k]

            # get sampling rate (the time_stamp is in microseconds for tobii and ms for eyelink)
            self.recording_fs = self.time_scale/(np.round(np.mean(np.diff(time_stamp))))
            

            if eye == 'left':
                list_diameter_nan = list_nan_left_diameter
            elif eye == 'right':
                list_diameter_nan = list_nan_right_diameter
            else:
                warnings.warn("You must chose an eye !")
            
            

            counter_nan_diam_left, counter_nan_diam_right = 0,0
            for i in range(len(list_nan_left_diameter)):
                if (list_nan_left_diameter[i] and True):
                    counter_nan_diam_left += 1
            for i in range(len(list_nan_right_diameter)):
                if (list_nan_right_diameter[i] and True):
                    counter_nan_diam_right += 1

            # Variable which will be useful for the function pick_best_eye (only related to diameter data)
            self.nb_diam_nan_left = counter_nan_diam_left
            self.nb_diam_nan_right = counter_nan_diam_right
            self.list_diameter_nan = list_diameter_nan
            if eye == 'left':
                nb_diameter_nan = counter_nan_diam_left
            elif eye == 'right':
                nb_diameter_nan = counter_nan_diam_right
            else:
                nb_diameter_nan = 0
                warnings.warn("You must chose an eye !")

            self.nan_ratio = nb_diameter_nan/len(list_diameter_nan)
            self.nb_diameter_nan = nb_diameter_nan

        


    def preprocess(self, eyetracker='tobii',margin=0.1, lpf=[20], plot=False, remove_outlier=False, outlier_limit_std=10, outlier_correct_average_amount=15):
        """
        Preprocess the data
            - eyetracker : use different preprocess according to the specified eyetracker
            - margin: added before and after each blink. Data during blinks is linearly interpolated
            - lpf: cutoff (Hz) to low pass filter the data with. If [], do nothing
            - plot: plot the result if True
            - remove_outlier is a function to replace the artifacts in the column diameter of the data.
                A data is considered as an atirfact if it's different from 0 and the derivative of the diameter is greater than
              'outlier_limit_std' x the standard deviation of the derivative
            - outlier_correct_average_amount is the number of data at the left and at the right of the artifact which will replace the value of 
              the artifact by the value of the mean of the data around
        The function gets the blinks (NB: currently from eyelink, but a custom function could be
        used) instead and linearly interpolate the data.
        For low pass filtering, a bi-directional 4th order Butterworth filter is used.
        The function replace previously pre-processed data, if any.
        It returns an indicator column in the panda dataframe indicating blink periods.
        """

        if eyetracker=='tobii':
            # INTERPOLATE BETWEEN BLINKS
            # get columns to process
            col_to_change = [
                col for col in self.data.columns if col != 'track_int']

            for col in col_to_change:
                self.data[col+'_int'] = copy.deepcopy(self.data[col])
            self.data['track_int'] = np.zeros(self.data.shape[0])
            time_stamp = np.array(self.data['time_stamp'])

            # merge blinks that are too close to one another
            is_long = (self.blink_start[1:] - self.blink_end[
                0:-1]) > margin * 2 * self.time_scale
            self.blink_start = self.blink_start[np.hstack((True, is_long))]

            self.blink_end = self.blink_end[np.hstack((is_long, True))]

            # interpolate the signal between blink
            first = self.data['time_stamp'].iloc[0]
            last = self.data['time_stamp'].iloc[-1]

            for start, end in zip(self.blink_start, self.blink_end):
                t = (time_stamp >= (start - margin * self.time_scale)) & \
                    (time_stamp <= (end + margin * self.time_scale))
                self.data.loc[t, 'track_int'] = 1
                for col in col_to_change:
                    val_start = self.data[col + '_int'][
                    np.where(t)[0][0]] if start > (first+1)\
                    else self.data[col+'_int'][np.where(t)[0][-1]]
                    val_end = self.data[col+'_int'][
                        np.where(t)[0][-1]] if end < last else val_start
                    self.data.loc[t, col+'_int'] = np.linspace(
                        val_start, val_end, np.sum(t)+1)[1:]

            # LOW PASS FILTER
            if lpf != []:
                w = lpf / (self.recording_fs / 2)  # Normalize the frequency
                b, a = signal.butter(4, w, 'low')  # 4-th order butterworth filter
                for col in col_to_change:
                    # use a bidirectional (0-lag) filter
                    self.data[col + '_int'] = signal.filtfilt(b, a, self.data[
                        col + '_int'])
                    

            # AVOID HUGE GAP IN DIAMETER : Replace artefact
            if remove_outlier:
                diameter_int = self.data['diameter_int']
                # derivative list of the diameter interpolated list
                rate_change_list = [] 

                for i in range(len(diameter_int)-1):
                    if (diameter_int[i+1] != 0 and diameter_int[i] != 0):
                        rate_change_list.append(abs(diameter_int[i+1]-diameter_int[i]))
                    else:
                        pass
                rate_std = stats.tstd(rate_change_list) 
                offset = 0 # if you want for any reason to move verticaly the interpolated diameter on your graph, keep value to  if you don't
                filtered_diameter = [diameter_int[0]+offset]
                for i in range(1,len(diameter_int)):
                    # An artefact is different from a NaN so we verified that data are different from 0 (but I'm not sure it's necessary cause artefact can be 0 too)
                    if (diameter_int[i-1] != 0 and diameter_int[i] != 0):
                        if abs(filtered_diameter[i-1]-diameter_int[i]) > outlier_limit_std*rate_std :
                            # if the artefact is at the beginning of data, it won't be possible to average at the right and left => only average of the right
                            if i < outlier_correct_average_amount:
                                mean_list = []
                                for j in list(range(1,outlier_correct_average_amount+1)):
                                    #print(" j = ", j)
                                    mean_list.append(diameter_int[i+j])
                                filtered_diameter.append(np.mean(mean_list)+offset)
                            # if the artefact is at the end of data, it won't be possible to average at the right and left => only average of the left
                            elif i > len(diameter_int) - outlier_correct_average_amount:
                                mean_list = []
                                for j in list(range(-outlier_correct_average_amount,0)):
                                    mean_list.append(filtered_diameter[i+j])
                                filtered_diameter.append(np.mean(mean_list)+offset)
                            else:
                                try:
                                    mean_list = []
        
                                    for j in list(range(-outlier_correct_average_amount,0)):
                                        mean_list.append(filtered_diameter[i+j])

                                    for j in list(range(1,outlier_correct_average_amount+1)):
                                        mean_list.append(diameter_int[i+j])
                                    filtered_diameter.append(np.mean(mean_list)+offset)
                                # Reduce the length of the average list if there's an error for any reason
                                except:
                                    while len(diameter_int)-i < outlier_correct_average_amount+1:
                                        outlier_correct_average_amount = outlier_correct_average_amount - 1
                                    mean_list = []

                                    for j in list(range(-outlier_correct_average_amount,0)):
                                        mean_list.append(filtered_diameter[i+j])

                                    for j in list(range(1,outlier_correct_average_amount+1)):
                                        mean_list.append(diameter_int[i+j])
                                    filtered_diameter.append(np.mean(mean_list)+offset)
                        else:
                            filtered_diameter.append(diameter_int[i]+offset)
                    else:
                        filtered_diameter.append(diameter_int[i]+offset)

                #Replace the interpolated diameter list by the list without the peaks considered as artefacts
                self.data['diameter_int'] = filtered_diameter


            #  PLOT TO CHECK QUALITY
            if plot:
                plt.figure()
                plt.title('diameter')
                plt.plot(self.data['time_stamp_rel_s'], self.data['diameter'], '-',
                        self.data[
                            'time_stamp_rel_s'], self.data['diameter_int'], '-')
                # plt.show()
                # plt.figure()
                # plt.title('x_pos')
                # plt.plot(self.data['time_stamp_rel_s'], self.data['x_pos'], '-',
                #          self.data[
                #              'time_stamp_rel_s'], self.data['x_pos_int'], '-')
                # plt.show()
                # plt.figure()
                # plt.title('y_pos')
                # plt.plot(self.data['time_stamp_rel_s'], self.data['y_pos'], '-',
                #          self.data[
                #              'time_stamp_rel_s'], self.data['y_pos_int'], '-')
                plt.show()
            return self.data['time_stamp_rel_s'], self.data['diameter'], self.data['diameter_int'], self.data['x_pos'], self.data['x_pos_int'], self.data['y_pos'], self.data['y_pos_int']
        
        elif eyetracker == 'eyelink':
            # INTERPOLATE BETWEEN BLINKS
            # get columns to process
            col_to_change = [col for col in self.data.columns if col != 'track_int']
            for col in col_to_change:
                self.data[col+'_int'] = copy.deepcopy(self.data[col])
            self.data['track_int'] = np.zeros(self.data.shape[0])
            time_stamp = np.array(self.data['time_stamp'])

            # merge blinks that are too close to one another
            is_long = (self.blink_start[1:] - self.blink_end[0:-1]) > margin*2*1000
            self.blink_start = self.blink_start[np.hstack((True, is_long))]
            self.blink_end = self.blink_end[np.hstack((is_long, True))]

            # interpolate the signal between blink
            first = self.data['time_stamp'].iloc[0]
            last = self.data['time_stamp'].iloc[-1]
            for start, end in zip(self.blink_start, self.blink_end):
                t = (time_stamp >= (start-margin*1000)) & (time_stamp <= (end+margin*1000))
                self.data.loc[t, 'track_int'] = 1
                for col in col_to_change:
                    val_start = self.data[col+'_int'][np.where(t)[0][0]] if start > (first+1)\
                        else self.data[col+'_int'][np.where(t)[0][-1]]
                    val_end = self.data[col+'_int'][np.where(t)[0][-1]] if end < last else val_start
                    self.data.loc[t, col+'_int'] = np.linspace(val_start, val_end, np.sum(t))

            # LOW PASS FILTER
            if lpf != []:
                w = lpf / (self.recording_fs / 2)  # Normalize the frequency
                b, a = signal.butter(4, w, 'low')  # 4-th order butterworth filter
                for col in col_to_change:
                    # use a bidirectional (0-lag) filter
                    self.data[col+'_int'] = signal.filtfilt(b, a, self.data[col+'_int'])

            # PLOT TO CHECK QUALITY
            if plot:
                plt.figure()
                plt.plot(self.data['time_stamp_rel_s'], self.data['diameter'], '-',
                        self.data['time_stamp_rel_s'], self.data['diameter_int'], '-')



    def epoch_tobii(self, onsets, before, after, variable='diameter_int',
                    decimate=1, reject=False, blink_ratio_thd=0.2,
                    conditions={}, baseline=[], z_score_data=False):
        """
        Epoch the data
            - onset: list of points around which to epoch. Onsets are according to the
                data['time_stamp']
            - before / after: duration (in s) before and after the onset. 
            - variable: which element of data.data should be epoched
            - decimate: downsampling factor (int). If 1, there is no downsampling
            - reject: remove epoch with an excessive blink ratio
            - blink_ratio: threshold above which the epoch is removed
            - conditions: dictionary which define properties of the onset (it will be copied, after
                rejecting the relevant epochs, in the output)
            - baseline: [start, end] in s relative to the onset, use for subtractive basline
                correction. If [], no correction is made
            - z_score: option to zscore the data before computing the epochs
                Apply a scipy z_score to the entier variable
        It returns:
            epochs: the curated, baseline-corrected epochs (as a numpy array, epoch x time)
            epochs_times: the peri-stimulus time vector (in s).
            epoch_blink_ratio: the author hasn't explained it..
            nan_ratio_epoch: the ratio of NaN per epoch in the column of the diameter of the picked eye
            epochs_info: a panda dataframe with the epoch number, the blink ratio, NaN ratio per epoch, and conditions (
                rejected epoch do not appear there)
                blink ratio and NaN ratio epoch should be the same but it's not. NaN ratio epoch do really count 
                the number of NaN per epoch and divid it by the length.
        """

        # Z_score the data on the whole session
        if z_score_data :
            self.data[variable] = scipy.stats.zscore(self.data[variable])

        epochs = []
        epoch_times = []
        blink_ratio = []
        nan_ratio_epoch = []

        # Variable to check the difference of epoch length (sometimes Tobii doesn't sample regularly)
        list_len_epochs = []

        #print(f" {len(onsets)} epochs counted, is it normal ?!")

        for onset in onsets:

            indices = (self.data['time_stamp'] >= onset-int(before*self.time_scale)) &\
                      (self.data['time_stamp'] <= onset+int(after*self.time_scale))
            
            chunk = self.data[variable].loc[indices]


            # Count the number of 'Nan' per epoch and count the number of NaN in a row
            True_counter = 0
            for i,j in zip(indices, self.list_diameter_nan):
                if i and j:
                    True_counter += 1
            nan_ratio_epoch.append(True_counter/len(chunk))

            # all the epoch and time arrays will have the size of the first epoch
            if onset == onsets[0]:
                epochs.append(chunk.values)
                epoch_times = (self.data['time_stamp'].loc[indices].values[0::decimate] - onset)/self.time_scale
            
            # Interpolate epochs to make them having all the same length (length of the 1st epoch)
            elif len(epochs[0]) != len(chunk.values):
                epoch = chunk.values
                time_value = (self.data['time_stamp'].loc[indices].values[0::decimate] - onset)/self.time_scale
                f = interp1d(time_value, epoch)
                x_new = np.linspace(time_value[0],time_value[-1],len(epoch_times))
                interpolate_epoch = f(x_new)
                epochs.append(interpolate_epoch)

            # # if the other epochs are bigger that the first one, they are shortcut
            # elif len(epochs[0]) < len(chunk.values):
            #     value = chunk.values[0:len(epochs[0])]
            #     epochs.append(value)

            # # if the other epochs are smaller that the first one, 0 are added (not my choice..)
            # elif len(epochs[0]) > len(chunk.values):
            #     value = np.concatenate([chunk.values,np.zeros(len(epochs[0]) - len(chunk.values))])
            #     epochs.append(value)

            else:
                epochs.append(chunk.values)
            list_len_epochs.append(len(chunk.values))
           
            blink_ratio.append(np.mean(self.data['track_int'].loc[indices]))

        epoch_blink_ratio = np.array(blink_ratio)
        epochs = np.vstack(epochs)[:, 0::decimate]

       
        if max(list_len_epochs) - min(list_len_epochs) > int(0.005*np.mean(list_len_epochs)):
            warnings.warn("Attention, il y a un important écart dans la taille des epochs, >0.5% !")
        if max(list_len_epochs) - min(list_len_epochs) > int(0.01*np.mean(list_len_epochs)): 
            warnings.warn(" The difference of length between epochs is too important (>1%), the procedure is to shutdown the program ")
            #print(1/0)


        # Function to define the baseline and substrate its mean to the epoch # Baseline correct
        baseline_val = []
        if baseline != []:
            baseline_ind = (epoch_times >= baseline[0]) &\
                (epoch_times <= baseline[1])
            baseline_val = np.mean(epochs[:, baseline_ind], axis=1)
            epochs = epochs - baseline_val[:, np.newaxis]
        else:
            baseline_val = np.zeros(epochs.shape[0])


        # reject epochs with too much blinks
        epoch_number = np.arange(0, epochs.shape[0])
        if reject is True:
            to_keep = epoch_blink_ratio <= blink_ratio_thd
            epochs = epochs[to_keep, :]
            epoch_number = epoch_number[to_keep]
            epoch_blink_ratio = epoch_blink_ratio[to_keep]
            baseline_val = baseline_val[to_keep]

            for info in conditions.keys():
                conditions[info] = np.array(conditions[info])[to_keep]


        # assemble the data
        self.epochs = epochs
        self.epochs_times = epoch_times
        self.conditions = conditions
        self.epoch_blink_ratio = epoch_blink_ratio
        self.nan_ratio_epoch = nan_ratio_epoch
        self.epochs_info = pd.DataFrame(data=dict(
            {'epoch_number': epoch_number,
             'blink_ratio': epoch_blink_ratio,
             'nan_ratio_epoch': nan_ratio_epoch,
             'baseline': baseline_val}, ** conditions))
        

    def epoch_eyelink(self, onsets, before=1, after=1, variable='diameter_int', decimate=1,
              reject=False, blink_ratio_thd=0.2, conditions={}, baseline=[], z_score_data=False):
        """
        Epoch the data
            - onset: list of points around which to epoch. Onsets are according to the
                data['time_stamp']
            - before / after: duration (in s) before and after the onset
            - variable: which element of data.data should be epoched
            - decimate: downsampling factor (int). If 1, there is no downsampling
            - reject: remove epoch with an excessive blink ratio
            - blink_ratio: threshold above which the epoch is removed
            - conditions: dictionary which define properties of the onset (it will be copied, after
                rejecting the relevant epochs, in the output)
            - baseline: [start, end] in s relative to the onset, use for subtractive basline
                correction. If [], no correction is made
        It returns:
            epochs: the curated, baseline-corrected epochs (as a numpy array, epoch x time)
            epochs_times: the peri-stimulus time vector (in s).
            epochs_info: a panda dataframe with the epoch number, the blink ratio, and conditions (
                rejected epoch do not appear there)
        """
        # Z_score the data on the whole session (Aaron addition)
        if z_score_data :
            self.data[variable] = scipy.stats.zscore(self.data[variable])
        
        # epoch the data
        epochs = []
        blink_ratio = []
        for onset in onsets:
            indices = (self.data['time_stamp'] >= onset-int(before*1000)) &\
                      (self.data['time_stamp'] <= onset+int(after*1000))
            chunk = self.data[variable].loc[indices]        
            if onset == onsets[0]:
                epochs.append(chunk.values)
            elif len(epochs[0]) < len(chunk.values):
                value = chunk.values[0:len(epochs[0])]   
                epochs.append(value)
            elif len(epochs[0]) > len(chunk.values):
                value = np.concatenate([chunk.values, 
                                        np.zeros(len(epochs[0]) - len(chunk.values))]) 
                epochs.append(value)
            else: 
                epochs.append(chunk.values)
            blink_ratio.append(np.mean(self.data['track_int'].loc[indices]))
        epoch_blink_ratio = np.array(blink_ratio)
        epochs = np.vstack(epochs)[:, 0::decimate]

        # get peri onset times
        epoch_times = (self.data['time_stamp'].loc[indices].values[0::decimate] - onset)/1000


        # baseline correct
        print(f"epochs shape: {epochs.shape}")  # Check the number of columns (Aaron addition)
        print(f"epoch_times length: {len(epoch_times)}")  # Check the length of epoch_times (Aaron addition)
  
        if epochs.shape[1] != len(epoch_times):
            if epochs.shape[1] > len(epoch_times): #(Aaron addition)
                epochs = epochs[:,0:-1] 
            elif epochs.shape[1] < len(epoch_times): #(Aaron addition)
                epoch_times = epoch_times[0:-1] #(Aaron addition)

        assert epochs.shape[1] == len(epoch_times), "Mismatch between epochs and epoch_times dimensions" # (Aaron addition)

        baseline_val = []
        if baseline != []:
            baseline_ind = (epoch_times >= baseline[0]) & (epoch_times <= baseline[1])
            baseline_val = np.mean(epochs[:, baseline_ind], axis=1)
            epochs = epochs - baseline_val[:, np.newaxis]
        else:
            baseline_val = np.zeros(epochs.shape[0])

        # reject epochs with too much blinks
        epoch_number = np.arange(0, epochs.shape[0])
        if reject is True:
            to_keep = epoch_blink_ratio <= blink_ratio_thd
            epochs = epochs[to_keep, :]
            epoch_number = epoch_number[to_keep]
            epoch_blink_ratio = epoch_blink_ratio[to_keep]
            baseline_val = baseline_val[to_keep]
            for info in conditions.keys():
                conditions[info] = conditions[info][to_keep]

        # assemble the data
        self.epochs = epochs
        self.epochs_times = epoch_times
        self.epochs_info = pd.DataFrame(data=dict({'epoch_number': epoch_number,
                                                   'blink_ratio': epoch_blink_ratio,
                                                   'baseline': baseline_val},
                                                  **conditions))

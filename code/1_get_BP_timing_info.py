# The purpose of this script is to extract trial timing information from the raw button-press MEG data in the CamCAM dataset, and write this information to a single csv file
# containing information (button press onset time, time since previous button press, and time until the next button press) for every trial and participant.


# Import packages
import logging
import csv
import os, sys
import logging
import pandas as pd
import numpy as np
import mne
from statistics import mode
import multiprocessing as mp
import tqdm
from pandas.errors import EmptyDataError

# Set up DIRs
data_path = os.path.join("/media/WDEasyStore/timb/camcan/download/20170824/cc700/meg/pipeline/release004/data_movecomp/aamod_meg_maxfilt_00002")
demographics_path = os.path.join("/home/timb/camcan/proc_data/")
out_path = os.path.join("/media/NAS/lbailey/PMBR_timecourse/output/")

log_out_path = os.path.join(out_path, 'examine_trails_logs')
if not os.path.exists(log_out_path):
    os.makedirs(log_out_path)

fig_out_path = os.path.join(out_path, 'figures')
if not os.path.exists(fig_out_path):
    os.makedirs(fig_out_path)

event_numpy_out_path = os.path.join(out_path, 'event_numpys')
if not os.path.exists(event_numpy_out_path):
    os.makedirs(event_numpy_out_path)

# Read csv file containing demographic info
df_demo = pd.read_csv(os.path.join(demographics_path, "demographics_allSubjects.csv"))

# Get list of subjects with existing raw file from the demographics dataframe
subject_list = list(df_demo.loc[(df_demo['RawExists'] == 1)]['SubjectID'])

# Create an empty csv to store timing data for all the button press trials
trial_timings_csv_fname = os.path.join(out_path, 'trial_timings.csv')

with open(trial_timings_csv_fname, 'w+', newline='') as outcsv:
    writer = csv.writer(outcsv)
    pass


# Setup logger
class StreamToLogger(object):
   """
   Fake file-like stream object that redirects writes to a logger instance.
   """
   def __init__(self, logger, log_level=logging.INFO):
      self.logger = logger
      self.log_level = log_level
      self.linebuf = ''

   def write(self, buf):
      for line in buf.rstrip().splitlines():
         self.logger.log(self.log_level, line.rstrip())

# Define function to pull the number of button-press events with a given set of ISI's
def examine_trials(subject, trial_timings_csv_fname=trial_timings_csv_fname):

    # First pull the rows from demo_df corresponding to this subject
    df_subject_demo = df_demo.loc[(df_demo['SubjectID'] == subject)][['SubjectID', 'Age', 'Gender', 'Hand']]
    df_subject_demo.rename(columns={'SubjectID': 'subject', 'Age': 'age', 'Gender': 'gender', 'Hand': 'hand'}, inplace=True)

    # Define path to raw file 
    raw_fif_fname = os.path.join(data_path, subject, 'task', 'transdef_transrest_mf2pt2_task_raw.fif')

    # Skip subjects without a raw file
    if not os.path.exists(raw_fif_fname):
        print(f'Raw file for subject {subject} does not exist. Skipping...')
        return
    

    # Setup log file - we'll just dump them in the scripts folder for now
    logFile = os.path.join(log_out_path, subject + '_ERB_processing_notes.txt')
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(message)s',
        filename=logFile,
        filemode='a'
    )
    stdout_logger = logging.getLogger('STDOUT')
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl
    stderr_logger = logging.getLogger('STDERR')
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl
    
    
    # Read in the raw data   
    raw = mne.io.read_raw_fif(raw_fif_fname, preload=False)

    # Read events (from all channels)
    events = mne.find_events(raw, uint_cast=True, shortest_event=1)

    # Save the raw event numpy as a txt file
    mne.write_events(os.path.join(event_numpy_out_path, subject + '-eve.txt'), events)

    # If we've already loaded the data once, we can get events much faster by simply loading the event numpys
    # events = mne.read_events(os.path.join(event_numpy_out_path, subject + '-eve.txt'))

    # Events in a Numpy array with event code in the third column. We want to subset only the button-press events (i.e., the most frequent event code)
    events_bp = events[events[:,2] == mode(events[:,2])]

    # Convert to pandas and assign sensible names to columns. We can drop the middle column. 
    df_bp = pd.DataFrame(events_bp[:,[0,2]], columns=['this_trial_t', 'event_code'])

    # Assign a new column wth time of the PREVIOUS trial (first row will be null)
    df_bp['prev_trial_t'] = df_bp['this_trial_t'].shift(1)

    # Do the same again, but with the time of the NEXT trial
    df_bp['next_trial_t'] = df_bp['this_trial_t'].shift(-1)

    # Create two new columns - time since last trial and time until next trial 
    df_bp['t_since_prev_trial'] = df_bp['this_trial_t'] - df_bp['prev_trial_t']
    df_bp['t_until_next_trial'] = df_bp['next_trial_t'] - df_bp['this_trial_t']

    # Convert to seconds
    df_bp = df_bp[df_bp.columns]/1000

    # But not the event code..
    df_bp['event_code'] = (df_bp['event_code'] * 1000).astype(int)

    # Append a new column for subject
    df_bp['subject'] = subject

    # Drop rows containing any NaNs
    df_bp.dropna(inplace=True)

    # Now merge df_bp with df_subject_demo
    df_bp_demo = df_subject_demo.merge(right=df_bp, on='subject')

    # Append the detailed trial timing info to a csv. Check whether the csv is empty with a try statement 
    # If it is, pd.read_csv will throw an EmptyDataError, in which case we'll add headers
    try: 
        pd.read_csv(trial_timings_csv_fname)
    except EmptyDataError:
        df_bp_demo.to_csv(trial_timings_csv_fname, mode='a', index=False, header=True)
    else:
        df_bp_demo.to_csv(trial_timings_csv_fname, mode='a', index=False, header=False)


# Run the function on all subjects in parallel
if __name__ == '__main__':

    # Set up the parallel task pool to use all available processors
    count = int(np.round(mp.cpu_count()*3/4))
    pool = mp.Pool(processes=count)

    # Display progress bar
    for _ in tqdm.tqdm(pool.imap_unordered(examine_trials, subject_list), total=len(subject_list)):
      pass

    # Run the jobs
    pool.map(examine_trials, subject_list)

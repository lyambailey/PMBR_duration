# Outline: The purpose of this script is to compute Bayes factor timecourses from the beta timecourses in each channel. 
# We will compute the Bayes factor for each time point in the beta timecourse, comparing the beta power at that time point  
# to the average beta power after 10 seconds.

# Import packages
library(BayesFactor)
library(reticulate)
library(ggplot2)
library(scales)
library(effsize)
np <- import('numpy') # This will trigger a prompt to set up a new environment. Enter "no"

# Define whether to use the default priors (with no nullInterval) or to use a nullInterval derrived from baseline data 
use_null_interval=TRUE

# Define number of iterations (if using null interval)
n_its = 1000

# Define path to data and output filenames
data_path <- '/media/NAS/lbailey/PMBR_timecourse/output/1BP15/' 
bp_data_fname = paste(data_path, 'tfr_data_BP_beta_617_subjects.npy', sep='')
bp_times_fname = paste(data_path, 'bp_times.npy', sep='')

if (use_null_interval) {
  out_suffix = paste('_nullInterval', n_its, 'iters', sep='')
} else {
  out_suffix=''
}

out_fname = paste(data_path, 'BFs_on_beta_timecourse', out_suffix, '.csv', sep='')

# Load beta power timecourses.npy
data_orig <- np$load(bp_data_fname) 

# Load times array
times <- np$load(bp_times_fname) 

# Define an empty array to store Bayes Factors
bf <- data.frame()

# Define empty vector to store the mean effect size at baseline for each channel (only used if use_null_interval=TRUE)
mean_ds = c()

# Loop through channels
for (channel in 1:2) {

  # Subset data for this channel
  data <- data_orig[, channel,]

  # Subset the baseline data (i.e., 10-15 seconds)
  baseline_data = data[,1100:ncol(data)]

  # Compute average baseline data across time, but preserve epochs
  baseline_data_mean_over_time = np$mean(baseline_data, axis=as.integer(1))

  # Stretch baseline_data_mean to the same length as data. Note that we transpose the output of np$tile here using t()
  baseline_data_mean_over_time_tiled = t(np$tile(baseline_data_mean_over_time, c(ncol(data), as.integer(1))))

  # Get nullInterval limits for the cauchy prior. We'll do this by iteratively splitting the baseline data 
  # (averaged over epochs) into random halves and computing an effect size (d). We'll take the mean effect size 
  # (mean_d) as the lower limit of the nullInterval parameter.  Note that, for simplicity, we perform this step whether 
  # or not use_null_interval is True - if False, we simply do not use mean_d when computing the Bayes Factors.

  # Define empty list to store effect sizes
  ds = c()

  # Average the baseline data pver epochs
  baseline_data_mean_over_epochs = np$mean(baseline_data, axis=as.integer(0))

  # Loop through iterations
  for (i in 1:n_its) {

    # Split the baseline data into two halves
    baseline_len = length(baseline_data_mean_over_epochs) 
    half_baseline_len = as.integer(baseline_len/2)

    # Generate a sequence of values between 1 and baseline_len, of length (baseline_len/2)
    # We'll use this to split the data into two halves
    idxs = sample(1:baseline_len, half_baseline_len)

    half1 = baseline_data_mean_over_epochs[idxs] 
    half2 = baseline_data_mean_over_epochs[-idxs] 

    # Compute effect size
    d = abs(cohen.d(half1, half2)$estimate)

    # Append effect size to list
    ds = c(ds, d)
  }

  # Compute mean effect size
  mean_d = mean(ds)

  # Add mean effect size to a list. Mean_ds should unltimately contain two values, one for each channel
  mean_ds = c(mean_ds, mean_d)

  # # Loop through columns of slopes and calculate Bayes Factor for each
  df <- as.data.frame(data)

  # We can either use the default nullInterval (NULL), or we could use the effect size computed above
  if (use_null_interval) {
    nullInterval = c(mean_d, Inf)
  } else {
    nullInterval = NULL
  }

  # Loop through columns (i.e., time points) of df, computing a Bayes factor for each
  for (i in 1:ncol(df)) {

    # Perform twosample test (each time point is compared to the same vector of average baseline values across time)
    res = ttestBF(df[,i], baseline_data_mean_over_time_tiled[,i], nullInterval=nullInterval, paired=TRUE)

    # Extract the Bayes Factor from the result and append it to the bf array. 
    # Note that, in bf, rows = timepoints, columns = channels 
    bf[i, as.integer(channel)] = as.vector(res)[[1]]

  } # columns loop
} # channel loop

# Round null intervals to 2 sig figs and print
mean_ds = signif(mean_ds, 2)
print(mean_ds)
print(paste('Mean effect size for MEG0211:', mean_ds[1]))
print(paste('Mean effect size for MEG1311:', mean_ds[2]))

# Rename columns - include the null interval in the column name if applicable
if use_null_interval {
  ch1_col_suffix = mean_ds[1]
  ch2_col_suffix = mean_ds[2]
} else {
  ch1_col_suffix = ''
  ch2_col_suffix = ''
}

colnames(bf) <- c(paste('BF_MEG0211_', ch1_col_suffix, sep=''),
                  paste('BF_MEG1311_', ch2_col_suffix, sep='')
                  )

# Add a column for times
bf$Time <- times

# Check everything looks good
summary(bf)

# Save bf to csv
write.csv(bf, out_fname, row.names=FALSE)

# Running ICA with multiple restarts

1. Append your dataset to PRECISE (make sure to add new metadata rows)
2. Subtract the reference condition
  a. If you have a WT Glucose M9 media condition, use yours
  b. If you do not have this condition, use control__wt_glc__1 and control__wt_glc__2
3. Save dataset (usually named log_tpm_norm.csv)
4. Copy the contents of this directory to a folder on NERSC
5. Copy your dataset to above directory
6. Open consistency.sh
  a. Update FILE
  b. Update TIME to (TOTAL # Samples)/10 + 10 minutes
  c. Update EMAIL
7. Run ./consistency.sh
8. When all runs are complete, run ./export.sh
9. Copy outputs.tar.gz to your local computer. Extract and continue with Juypyter notebooks.

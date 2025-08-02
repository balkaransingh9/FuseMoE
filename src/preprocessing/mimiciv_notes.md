# Notes on MIMIC-IV data/processing/etc.

### Data
Tasks we discussed:

| IHM using the first 48 hours of data for patients who spent at least 48 hours in the ICU |  |
| --- | --- |
| Selection criteria | length-of-stay (in the ICU) >- 48 hours |
| Data included | first 48 hours of vitals/labs |
| Data files | `*_ihm-48_stays.pkl` where `*` is either `train`, `val`, or `test` |
| Running the task | set `--task` to `ihm-48` in `run_mimiciv.sh` |

| IHM with no restrictions |  |
| --- | --- |
| Selection criteria | Any ICU stay of any length |
| Data included | All labs/vitals for the ICU stay |
| Data files | `*_ihm-all_stays.pkl` |
| Running the task | set `--task` to `ihm-all` in `run_mimiciv.sh` |

Each of the pickle files contains time series (labs + vitals that were included in the NPJ paper) and radiological notes for the specified time frame (within 48 hours for `ihm-48` or at any point during ICU stay for `all-stays`). Additionally, only patients for whom there is *at least* one lab/vital and *at least* one radiological note within the specified time point are included.

Additionally, `--file_path` in `run_mimiciv.sh` should link the folder that contains the `.pkl` files.

### LOS task
In this task, I create a variation on the LOS task used in Soenkensen et al. (2023). The idea is basically to use a similar template for the IHM task -- for pts who stayed in the ICU for *at least 48 hours*, we try to predict discharge within the following *48 hours* (i.e., discharge within 96 hours in total) using (only) data from the first 48 hours. Specifically, it's a binary classification task, where the goal is to predict discharge from the ICU without expiration (death) within 48 hours. The label is `1` if the pt was discharged (alive) within 96 hours, and `0` if the patient stayed in the ICU for more than 96 hours (48 hours for the data window + 48 hours for the prediction interval) *or* if the patient died.

Data is contained in the Google Drive under `mimiciv_data/12-29-23`. The files **without** missingness indicators are saved under `*_los-48-cxr-notes_stays.pkl`, and those **with** missingness indicators are stored under `*_los-48-cxr-notes-missingInd_stays.pkl`, where `*` is in (`train`, `val`, `test`).

The sample sizes for the LOS task are the same as for IHM -- 8,770 for the files without missingness indicators, and 35,131 for the files with missingness indicators.

### CXR embeddings
CXR embeddings are contained in `.pkl` files that include `-cxr-`. The files contain pts for which we have both at least 1 CXR, at least 1 clinical note, and time series measurements in the specified time frame (48 hrs after admission or any time during admission, depending on the file). To run using the files, specify `task='*-48-cxr-notes'` or `task='*-all-cxr-notes'` (where `*` is `ihm` or `pheno`). Then, to use time series & notes specify `modeltype='TS_CXR'` (you can also do `modeltype='TS'` or `modeltype='TS_Text'` to run the time series or time series + text tasks on the sample). Should should also specify `--irregular_learn_emb_cxr` in your input args to learn the irregular time embedding for CXRs.

### MISSING DATA
I've uploaded new versions of the 48-IHM and 25-pheno task to the Google Drive (under `mimiciv_data/12-26-23`). These files are called `*_?-cxr-notes-missingInd_stays.pkl` where `*` is in (train, test, val) and `?` in (`ihm-48` or `pheno-all`). Here, `missingInd` denotes that all relevant ICU stays (i.e., ICU stays with length-of-stay > 48 hrs, in the case of `ihm-48`, or all ICU stays, in the case of `pheno-all`) are included, and missing indicators are provided for pts with missing modalities.

Specifically, if a pt is missing notes during the stay, `text_data` and `text_time_to_end` are empty (`[]`), and the corresponding missing indicator `text_missing = 1`. If the pt's observations are *not* missing, `text_missing=0`, `text_data` contains the text from the pt's notes, and `text_time_to_end` contains the timestamps of the note observations contained in `text_data`. 

Similarly, for CXRs, if an observation is missing `cxr_feats` and `cxr_time` are empty (`[]`) and `cxr_missing=1`. If the pt had *at least one* CXR, the `cxr_feats` contains the densenet embeddings, `cxr_time` the corresponding times at which each element of `cxr_feats` was recorded, and `cxr_missing=0`.

### 24-phenotyping
To run this task, specify `task='pheno-all-cxr-notes'` and `--num_labels 25` (for the 25 different acute conditions).

### NPJ Paper
To run this, you should download `icu_notes_text_embeddings.pkl` and `ts_labs_vitals_icu.pkl`, which contain the radiological notes and labs/vitals corresponding to all ICU stays. Then, you can generate embeddings and create predictions by running `npj_replication.ipynb` in the `npj` folder. This notebook contains the code to create TS embeddings according to the NPJ paper, and train/evaluate XGB models, using the code from the HAIM repository.


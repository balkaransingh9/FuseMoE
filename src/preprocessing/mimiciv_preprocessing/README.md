# Information about preprocessing notebooks & NPJ replication


## General notes
* `icu_time_delta` refers to the time difference (in hrs.) since the pt was admitted to the ICU (with the corresponding ICU stay denoted by `stay_id`).
* `hosp_time_delta` refers to the time difference (in hrs.) since the pt was admitted to the hospital (with the corresponding hospital stay denoted by `hadm_id`).

In the event the given measurements/CXR/etc. was not taken within the interval of a hospital/ICU stay, the corresponding `*_time_delta` is NaN, as is the corresponding `hadm_id` or `stay_id`. (I.e., if a pt stayed in the hospital but not the ICU, they will have a `hadm_id`/`hosp_time_delta` by no `stay_id`/`icu_time_delta`). Also note that, in the event a pt was admitted to the ICU, `icu_time_delta` is generally < `hosp_time_delta` for a given event, since pts usually enter the ICU after they enter the hospital.

## NOTEBOOKS
Notebooks contained in this `npj` directory:

### NPJ
* `npj_replication`: code to replicate Soenkensen et al. feature extraction + classifier (XGboost) pipeline.

### TS
* `ts_irregular`: extract chart (vitals)/lab events occuring during the ICU stay.
* `ts_imputed`: imputes TS values at regular, 1 hr. intervals, in accordinace with the imputation strategy from Zhang et al. (ICML, 2023).

### EMBEDDINGS
* `img_embeddings`: extract image embeddings (dense features/predictions) from each CXR jpg image in the `mimiciv-cxr` repository.
* `notes_text`: extracts the radiological reports text/metadata (hospital/ICU stay and hospital/icu). Saved to `notes_text.pkl`.
* `notes_embeddings` extracts text embeddings **only from radiological reports taken during an ICU stay** using the pretrained `bert_pretrain_output_all_notes_150000` model [from MIT](https://github.com/EmilyAlsentzer/clinicalBERT). Saved to `icu_notes_text_embeddings.pkl`.

### TASKS
* `create_ihm_task`: creates IHM task.
* `create_phenotyping_task`: Creates phenotyping task, as described in Harutyunyan et al. ([code](https://github.com/YerevaNN/mimic3-benchmarks/tree/master), [paper](https://www.nature.com/articles/s41597-019-0103-9)). Slightly complication in that the original paper only uses ICD-9 codes (which are used in MIMIC-III), whereas MIMIC-IV includes both ICD-9 *and* ICD-10 codes. This requires mapping relevant ICD-10 codes to their corresponding diagnosis-related group; I used `icd10cmtoicd9gem.csv` (from the [NBER](https://www.nber.org/research/data/icd-9-cm-and-icd-10-cm-and-icd-10-pcs-crosswalk-or-general-equivalence-mappings)) to do this mapping.

### MISC
* `plot_ts` plots vitals/time series (useful for visualizing sparsity)

## SCRIPTS
* `MIMIC_IV_HAIM_API.py`: api released for the Soenkensen et al. paper in NPJ.
* `prediction_util.py`: contains utility functions for XGBoost model fitting in the NPJ paper.
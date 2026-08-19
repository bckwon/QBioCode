#!/usr/bin/env bash
#==============================================================================
# LSF Job Array: QSVC Rerun — 48 missing train_qml.csv tasks (job_02g3)
#
# Job 75734 completed all 66 tasks but only produced train_qml.csv results for
# 17 of them. The remaining tasks' QSVC kernel fits were still in progress when
# their nodes became saturated. This array covers all 48 missing tasks:
#   - 45 that were never running
#   - 3 AMES tasks (75734[61,62,63]) killed after 12h silence on c08-04 @98%
#
# Config: admet_qsvc_config.yaml (same as job 75734)
#   - model: ['qsvc'] only
#   - embeddings: ['pca', 'umap']
#   - C: 1.0, MinMaxScaler fix applied
#
# Results append to results/admet_qsvc_config/ (new Hydra timestamps, no conflict)
# Merge script deduplicates on (folder_path, split) keeping newest — safe.
#
# Node selection: b05-02 (3% load), c06-01 (100% — will free as 76361/76362 finish),
#   c08-02 (13% load, newly available).
#
# Submit from repo root:
#   bsub < experiments/admet_benchmark/lsf/job_02g3_qsvc_missing45.sh
#==============================================================================

#BSUB -J admet_qsvc_g3[1-48]
#BSUB -q normal
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4000]"
#BSUB -m "zu-a100-b05-02 zu-a100-c06-01 zu-a100-c08-02"
#BSUB -gpu "num=1:mode=shared:j_exclusive=no"
#BSUB -o logs/admet/qsvc_g3_%I_%J.out
#BSUB -e logs/admet/qsvc_g3_%I_%J.err
#BSUB -cwd /proj/bmfm/users/bckwon/projects/QBioCode
#BSUB -r
#BSUB -nr 2

set -euo pipefail

REPO_ROOT="/proj/bmfm/users/bckwon/projects/QBioCode"
LOG_DIR="${REPO_ROOT}/logs/admet"
DATA_DIR="${REPO_ROOT}/data/admet"
CONFIG="${REPO_ROOT}/qbiocode/apps/qprofiler/configs/admet_qsvc_config.yaml"

mkdir -p "${LOG_DIR}"

# 48 missing tasks — 45 never-ran + 3 AMES (killed after 12h freeze on c08-04)
TASKS=(
    ""                                                   #  0 unused
    "Caco2_Wang:::ecfp4"                                 #  1  (75734[1])
    "HIA_Hou:::ecfp4"                                    #  2  (75734[4])
    "HIA_Hou:::maccs"                                    #  3  (75734[5])
    "HIA_Hou:::rdkit200"                                 #  4  (75734[6])
    "Pgp_Broccatelli:::maccs"                            #  5  (75734[8])
    "Bioavailability_Ma:::maccs"                         #  6  (75734[11])
    "Bioavailability_Ma:::rdkit200"                      #  7  (75734[12])
    "Lipophilicity_AstraZeneca:::ecfp4"                  #  8  (75734[13])
    "Lipophilicity_AstraZeneca:::maccs"                  #  9  (75734[14])
    "Lipophilicity_AstraZeneca:::rdkit200"               # 10  (75734[15])
    "BBB_Martins:::ecfp4"                                # 11  (75734[19])
    "BBB_Martins:::maccs"                                # 12  (75734[20])
    "PPBR_AstraZeneca:::ecfp4"                           # 13  (75734[22])
    "PPBR_AstraZeneca:::maccs"                           # 14  (75734[23])
    "PPBR_AstraZeneca:::rdkit200"                        # 15  (75734[24])
    "VDss_Lombardo:::ecfp4"                              # 16  (75734[25])
    "VDss_Lombardo:::maccs"                              # 17  (75734[26])
    "VDss_Lombardo:::rdkit200"                           # 18  (75734[27])
    "CYP2C19_Veith:::ecfp4"                              # 19  (75734[28])
    "CYP2C19_Veith:::maccs"                              # 20  (75734[29])
    "CYP2D6_Veith:::ecfp4"                               # 21  (75734[31])
    "CYP2D6_Veith:::rdkit200"                            # 22  (75734[33])
    "CYP3A4_Veith:::ecfp4"                               # 23  (75734[34])
    "CYP3A4_Veith:::maccs"                               # 24  (75734[35])
    "CYP3A4_Veith:::rdkit200"                            # 25  (75734[36])
    "CYP1A2_Veith:::ecfp4"                               # 26  (75734[37])
    "CYP1A2_Veith:::maccs"                               # 27  (75734[38])
    "CYP1A2_Veith:::rdkit200"                            # 28  (75734[39])
    "CYP2C9_Veith:::ecfp4"                               # 29  (75734[40])
    "CYP2C9_Substrate_CarbonMangels:::maccs"             # 30  (75734[44])
    "CYP2D6_Substrate_CarbonMangels:::ecfp4"             # 31  (75734[46])
    "CYP2D6_Substrate_CarbonMangels:::maccs"             # 32  (75734[47])
    "CYP3A4_Substrate_CarbonMangels:::ecfp4"             # 33  (75734[49])
    "CYP3A4_Substrate_CarbonMangels:::rdkit200"          # 34  (75734[51])
    "Half_Life_Obach:::maccs"                            # 35  (75734[53])
    "Half_Life_Obach:::rdkit200"                         # 36  (75734[54])
    "Clearance_Hepatocyte_AZ:::ecfp4"                    # 37  (75734[55])
    "Clearance_Hepatocyte_AZ:::maccs"                    # 38  (75734[56])
    "Clearance_Hepatocyte_AZ:::rdkit200"                 # 39  (75734[57])
    "hERG:::ecfp4"                                       # 40  (75734[58])
    "hERG:::maccs"                                       # 41  (75734[59])
    "hERG:::rdkit200"                                    # 42  (75734[60])
    "DILI:::ecfp4"                                       # 43  (75734[64])
    "DILI:::maccs"                                       # 44  (75734[65])
    "DILI:::rdkit200"                                    # 45  (75734[66])
    "AMES:::ecfp4"                                       # 46  (75734[61] killed)
    "AMES:::maccs"                                       # 47  (75734[62] killed)
    "AMES:::rdkit200"                                    # 48  (75734[63] killed)
)

TASK="${TASKS[${LSB_JOBINDEX}]}"
ENDPOINT="${TASK%%:::*}"
FEAT="${TASK##*:::}"

INPUT_DIR="${DATA_DIR}/${ENDPOINT}/${FEAT}"
DATA_TYPE="${ENDPOINT}_${FEAT}"

echo "======================================================"
echo "Job ID       : ${LSB_JOBID:-local}[${LSB_JOBINDEX:-1}]"
echo "Host         : $(hostname)"
echo "Start time   : $(date)"
echo "Task index   : ${LSB_JOBINDEX} / 45"
echo "Endpoint     : ${ENDPOINT}"
echo "Featurizer   : ${FEAT}"
echo "Input dir    : ${INPUT_DIR}"
echo "Config       : ${CONFIG}"
echo "======================================================"
echo "Fix applied  : ZZFeatureMap MinMaxScaler(0,1) + C=1.0"
echo "Models       : qsvc only"
echo "Embeddings   : pca, umap"
echo "======================================================"

if [ ! -d "${INPUT_DIR}" ]; then
    echo "[ERROR] Input directory not found: ${INPUT_DIR}"
    exit 1
fi

if [ ! -f "${INPUT_DIR}/train_qml.csv" ]; then
    echo "[ERROR] train_qml.csv not found in ${INPUT_DIR}"
    exit 1
fi

cd "${REPO_ROOT}"

PATH="${REPO_ROOT}/.venv/bin:${PATH}"
PYTHONPATH="${REPO_ROOT}" \
    qprofiler-batch \
    --input-dir  "${INPUT_DIR}" \
    --config     "${CONFIG}" \
    --data-type  "${DATA_TYPE}" \
    --n-jobs     2 \
    2>&1 | tee "${LOG_DIR}/qsvc_g3_${ENDPOINT}_${FEAT}_${LSB_JOBID:-local}.log"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "======================================================"
echo "End time  : $(date)"
echo "Exit code : ${EXIT_CODE}"
echo "======================================================"
exit ${EXIT_CODE}

DATASET_BIG_BENCH_MISTAKE = "big-bench-mistake"
DATASET_PRM800K = "prm800k"
DATASET_MR_GSM8K_ORIGINAL = "mr-gsm8k-original"
DATASET_ALL = "all"

SUPPORTED_DATASETS = (
    DATASET_BIG_BENCH_MISTAKE,
    DATASET_PRM800K,
    DATASET_MR_GSM8K_ORIGINAL,
)

PAPER_DATASETS = (
    DATASET_BIG_BENCH_MISTAKE,
    DATASET_PRM800K,
)

STAGE_1 = "stage1"
STAGE_2 = "stage2"
STAGE_2_STEP_TYPE = "stage2_step_type"
STAGE_2_REVIEW = "stage2_review"
STAGE_2_SPECIALIST_REVIEW = "stage2_specialist_review"

PRINCIPLE_LABELS = (
    "correct-and-aligned",
    "reasonable-but-incomplete",
    "nothing-extracted",
    "contradiction-found",
)

POSITIVE_TRACE_LABEL = 1
NEGATIVE_TRACE_LABEL = 0

"""Leakage control register — documented controls for every leakage type."""

LEAKAGE_REGISTER = [
    {
        "type": "Direct label leakage",
        "example": "churned passed as a feature",
        "control": "Explicit feature allowlist",
    },
    {
        "type": "Treatment misuse",
        "example": "contacted used as ordinary predictor",
        "control": "Use only as treatment variable",
    },
    {
        "type": "Temporal leakage",
        "example": "Usage after decision day",
        "control": "Point-in-time filter event_time <= decision_time",
    },
    {
        "type": "Outcome-window leakage",
        "example": "Different churn observation periods",
        "control": "Fixed label horizon",
    },
    {
        "type": "Preprocessing leakage",
        "example": "Scaling before split",
        "control": "Fit transformer in each fold",
    },
    {
        "type": "Target encoding leakage",
        "example": "Package encoding uses validation labels",
        "control": "Fold-aware encoding or CatBoost ordered encoding",
    },
    {"type": "Imputation leakage", "example": "Global medians", "control": "Train-fold medians"},
    {
        "type": "PCA/SVD leakage",
        "example": "Fit using all 60-day sequences",
        "control": "Fit only on training fold",
    },
    {
        "type": "Feature-selection leakage",
        "example": "Select features using all labels",
        "control": "Select inside inner CV",
    },
    {
        "type": "Oversampling leakage",
        "example": "SMOTE before splitting",
        "control": "Resample training fold only",
    },
    {
        "type": "Calibration leakage",
        "example": "Calibrate using test predictions",
        "control": "Separate OOF calibration",
    },
    {
        "type": "AutoML leakage",
        "example": "AutoML sees outer validation",
        "control": "AutoML only inside outer train",
    },
    {
        "type": "Stacking leakage",
        "example": "Meta-model uses in-sample base predictions",
        "control": "OOF base predictions only",
    },
    {
        "type": "Ranking leakage",
        "example": "Ranking customers using fitted outcomes",
        "control": "Cross-fitted ranking",
    },
    {
        "type": "Segment normalization leakage",
        "example": "Peer percentile calculated globally",
        "control": "Fit peer distributions on training data",
    },
    {
        "type": "Deduplication leakage",
        "example": "Same company in train and test",
        "control": "Group split if repeated companies exist",
    },
    {
        "type": "Eligibility leakage",
        "example": "Filtering based on future churn",
        "control": "Eligibility determined at decision time",
    },
    {
        "type": "Human-policy leakage",
        "example": "Account-manager notes unavailable at inference",
        "control": "Include only reproducible features",
    },
]


def get_leakage_register() -> list[dict]:
    return LEAKAGE_REGISTER

"""
Dataset definitions for GS-RVFL experiments.
"""

# Tabular datasets (15 datasets)
TABULAR_DATASETS = {
    'iris': {'samples': 150, 'features': 4, 'classes': 3},
    'wine': {'samples': 178, 'features': 13, 'classes': 3},
    'breast_cancer': {'samples': 569, 'features': 30, 'classes': 2},
    'digits': {'samples': 1797, 'features': 64, 'classes': 10},
    'mnist': {'samples': 5000, 'features': 784, 'classes': 10},
    'letter': {'samples': 20000, 'features': 16, 'classes': 26},
    'satellite': {'samples': 6435, 'features': 36, 'classes': 6},
    'pendigits': {'samples': 10992, 'features': 16, 'classes': 10},
    'optdigits': {'samples': 5620, 'features': 64, 'classes': 10},
    'waveform': {'samples': 5000, 'features': 21, 'classes': 3},
    'spambase': {'samples': 4601, 'features': 57, 'classes': 2},
    'adult': {'samples': 48842, 'features': 14, 'classes': 2},
    'covertype': {'samples': 10000, 'features': 54, 'classes': 7},
    'segment': {'samples': 2310, 'features': 19, 'classes': 7},
    'usps': {'samples': 9298, 'features': 256, 'classes': 10}
}

# Sensor datasets (3 datasets)
SENSOR_DATASETS = {
    'har': {'samples': 10299, 'features': 561, 'classes': 6},
    'wisdm': {'samples': 5000, 'features': 36, 'classes': 6},
    'pamap2': {'samples': 5000, 'features': 17, 'classes': 12}
}

# Video datasets (5 datasets)
VIDEO_DATASETS = {
    'kth': {'samples': 599, 'classes': 6},
    'weizmann': {'samples': 90, 'classes': 10},
    'ucf11': {'samples': 1600, 'classes': 11},
    'ucf101': {'samples': 2000, 'classes': 10},
    'hmdb51': {'samples': 1000, 'classes': 10}
}

# Skeleton datasets (2 datasets)
SKELETON_DATASETS = {
    'ntu': {'samples': 10000, 'classes': 60},
    'ntu120': {'samples': 10000, 'classes': 120}
}

# All datasets
ALL_DATASETS = {
    'tabular': TABULAR_DATASETS,
    'sensor': SENSOR_DATASETS,
    'video': VIDEO_DATASETS,
    'skeleton': SKELETON_DATASETS
}

# Domain-specific datasets (where GS-RVFL differs from RVFL)
DOMAIN_SPECIFIC_DATASETS = {
    'sensor': list(SENSOR_DATASETS.keys()),
    'video': list(VIDEO_DATASETS.keys()),
    'skeleton': list(SKELETON_DATASETS.keys())
}

# All domain-specific datasets
ALL_DOMAIN_SPECIFIC = (
    list(SENSOR_DATASETS.keys()) +
    list(VIDEO_DATASETS.keys()) +
    list(SKELETON_DATASETS.keys())
)

# Algorithm names for comparison
ALGORITHMS = [
    'GS-RVFL',
    'BLS+Attention',
    'BLS',
    'Deep RVFL',
    'Standard RVFL',
    'Multi-Layer RVFL',
    'Ensemble RVFL',
    'XGBoost',
    'Random Forest',
    'ELM'
]

# Baseline algorithms
BASELINES = {
    'classical': ['LogisticRegression', 'SVM_RBF', 'RandomForest', 'XGBoost'],
    'randomized': ['ELM', 'RVFL', 'RVFL_NoDirect', 'DeepRVFL', 'MultiLayerRVFL', 'EnsembleRVFL', 'BLS'],
    'domain': ['TwoStreamCNN', 'I3D', 'STGCN', 'CNN_LSTM']
}
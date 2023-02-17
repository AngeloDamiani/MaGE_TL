DATA_DIR = 'data/'
PROBLEMS_CONFIG = {
    'H' : 300,
    'MC': {
        'csvs': {
            'source': DATA_DIR+'UntrainedMCDataset5000.csv',
            'target': DATA_DIR+'UntrainedMCDataset500.csv',
            'source_val': DATA_DIR+'UntrainedMCDataset5000_2.csv',
            'target_val': DATA_DIR+'UntrainedMCDataset5000.csv',
            'test': DATA_DIR+'UntrainedMCDataset50000.csv' 
        }
    },
    'Pend': {
        'csvs': {
            'source': DATA_DIR+'UntrainedPendDataset5000.csv',
            'target': DATA_DIR+'UntrainedPendDataset500.csv',
            'source_val': DATA_DIR+'UntrainedPendDataset5000_2.csv',
            'target_val': DATA_DIR+'UntrainedPendDataset5000.csv',
            'test': DATA_DIR+'UntrainedPendDataset50000.csv' 
        }
    }
}

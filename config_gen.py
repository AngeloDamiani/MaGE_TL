from random import randint, uniform


def get_explore_config(id, prefix):
    return {
        "D_batch_size_pretrain": randint(100, 350),
        "D_epochs_pretrain": randint(20, 25),
        "D_batch_size": randint(140, 220),
        "D_lr": uniform(0.001, 0.003),
        "D_epochs": randint(10, 20),
        "AE_batch_size": randint(80, 450),
        "AE_lr": uniform(0.001, 0.006),
        "AE_epochs": randint(30, 95),
        "iterations": 20,
        "T_epochs": 100,
        "T_batch_size": 100,
        "T_lr": 0.0001,
        "lmd_D": uniform(4, 6),
        "lmd_T": uniform(0.5, 2.5),  # 1 - 10
        "lmd_R": uniform(11, 13),  # 1 - 10
        "log_dir": f"./logs/{prefix}_{id}",  # OK
    }

def get_random_config(id, prefix):
    return {
        "D_batch_size_pretrain": randint(50, 350),
        "D_epochs_pretrain": randint(10, 50),
        "D_batch_size": randint(50, 300),
        "D_lr": uniform(0.0001, 0.005),
        "D_epochs": randint(10, 30),
        "AE_batch_size": randint(20, 450),
        "AE_lr": uniform(0.0001, 0.005),
        "AE_epochs": randint(30, 95),
        "iterations": 20,
        "T_epochs": 100,
        "T_batch_size": 100,
        "T_lr": 0.0001,
        "lmd_D": uniform(0, 10),
        "lmd_T": uniform(0, 10),  # 1 - 10
        "lmd_R": uniform(0, 10),  # 1 - 10
        "log_dir": f"./logs/{prefix}_{id}",  # OK
    }


def get_ran1003_config(id, prefix):
    return {
    "D_batch_size_pretrain": 188,
    "D_epochs_pretrain": 25,
    "D_batch_size": 276,
    "D_lr": 0.0003732592074541112,
    "D_epochs": 30,
    "AE_batch_size": 88,
    "AE_lr": 0.0038688422199228494,
    "AE_epochs": 76,
    "iterations": 20,
    "T_epochs": 100,
    "T_batch_size": 100,
    "T_lr": 0.0001,
    "lmd_D": 1.7117454788938802,
    "lmd_T": 8.504341235671774,
    "lmd_R": 9.318351086808468,
        "log_dir": f"./logs/{prefix}_{id}",
    }


def get_g1_config(id, prefix):
    return {
        "D_batch_size_pretrain": 281,
        "D_epochs_pretrain": 25,
        "D_batch_size": 211,
        "D_lr": 0.002709221901091384,
        "D_epochs": 20,
        "AE_batch_size": 159,
        "AE_lr": 0.00299824264837295,
        "AE_epochs": 54,
        "iterations": 20,
        "T_epochs": 100,
        "T_batch_size": 100,
        "T_lr": 0.0001,
        "lmd_D": 5.492851091352457,
        "lmd_T": 1.531975708267081,
        "lmd_R": 12.55524708578032,
        "log_dir": f"./logs/{prefix}_{id}",
    }


def get_g4_config(id, prefix):
    return {
        "D_batch_size_pretrain": 215,
        "D_epochs_pretrain": 22,
        "D_batch_size": 156,
        "D_lr": 0.0025666759768773525,
        "D_epochs": 16,
        "AE_batch_size": 430,
        "AE_lr": 0.0017297598020690587,
        "AE_epochs": 70,
        "iterations": 20,
        "T_epochs": 100,
        "T_batch_size": 100,
        "T_lr": 0.0001,
        "lmd_D": 5.536489657352626,
        "lmd_T": 0.5385697817555073,
        "lmd_R": 12.518822822484182,
        "log_dir": f"./logs/{prefix}_{id}",
    }


def get_g5_config(id, prefix):
    return {
        "D_batch_size_pretrain": 215,
        "D_epochs_pretrain": 20,
        "D_batch_size": 159,
        "D_lr": 0.0024228473455324123,
        "D_epochs": 17,
        "AE_batch_size": 121,
        "AE_lr": 0.0029647678906922444,
        "AE_epochs": 80,
        "iterations": 20,
        "T_epochs": 100,
        "T_batch_size": 100,
        "T_lr": 0.0001,
        "lmd_D": 4.930014959059831,
        "lmd_T": 1.3742280719800923,
        "lmd_R": 12.507150274192803,
        "log_dir": f"./logs/{prefix}_{id}",
    }


def get_a1_config(id, prefix):
    return {
        "log_dir": f"./logs/{prefix}_{id}",
  "D_batch_size_pretrain": 308,
  "D_epochs_pretrain": 22,
  "D_batch_size": 189,
  "D_lr": 0.001172294006946692,
  "D_epochs": 19,
  "AE_batch_size": 346,
  "AE_lr": 0.0020911086155821104,
  "AE_epochs": 52,
  "iterations": 20,
  "T_epochs": 100,
  "T_batch_size": 100,
  "T_lr": 0.0001,
  "lmd_D": 5.108571768295475,
  "lmd_T": 0.8972927444857055,
  "lmd_R": 12.745102985142589
}

def get_default_config(id, prefix):
    return {
        "log_dir": f"./logs/{prefix}_{id}",
        "T_epochs": 100,
        "T_batch_size": 100,
        "T_lr": 0.001,
        "D_batch_size_pretrain": 237,
        "D_epochs_pretrain": 23,
        "D_batch_size": 176,
        "D_lr": 0.002566248,
        "D_epochs": 18,
        "AE_batch_size": 237,
        "AE_lr": 0.0025642567803780846,
        "AE_epochs": 68,
        "iterations": 20,
        "lmd_D": 5.319785235921638,
        "lmd_T": 1.1482578540008934,
        "lmd_R": 12.527073394152433,
    }


def get_debug_config(id, prefix):
    return {
        "log_dir": f"./logs/{prefix}_{id}",
        "T_epochs": 1,
        "T_batch_size": 1000,
        "T_lr": 0.001,
        "D_batch_size_pretrain": 1000,
        "D_epochs_pretrain": 1,
        "D_batch_size": 1000,
        "D_lr": 0.001,
        "D_epochs": 1,
        "AE_batch_size": 1000,
        "AE_lr": 0.001,
        "AE_epochs": 1,
        "iterations": 1,
        "lmd_D": 1,
        "lmd_T": 1,
        "lmd_R": 1,
    }

def get_ang0_config(id, prefix):
    return {
        "log_dir": f"./logs/{prefix}_{id}",
        "T_epochs": 100,
        "T_batch_size": 100,
        "T_lr": 0.001,
        "D_batch_size_pretrain": 237,
        "D_epochs_pretrain": 23,
        "D_batch_size": 176,
        "D_lr": 0.002566248,
        "D_epochs": 18,
        "AE_batch_size": 237,
        "AE_lr": 0.0025642567803780846,
        "AE_epochs": 68,
        "iterations": 20,
        "lmd_D": 5.319785235921638,
        "lmd_T": 1.1482578540008934,
        "lmd_R": 12.527073394152433
    }


config_functs = {
    "debug": get_debug_config,
    "default": get_default_config,
    "explore": get_explore_config,
    "G1": get_g1_config,
    "G4": get_g4_config,
    "G5": get_g5_config,
    "A1": get_a1_config,
    "ANG0": get_ang0_config,
    "RAN1003": get_ran1003_config,
}


def get_config(id, prefix="run", mode=None):
    config = config_functs.get(mode, get_random_config)  
    return config(id, prefix)

from mapping.datasets import RLDatasetFormatter
import mapping.config as cfg
import torch

def problem_dispenser(prob_name, dataset_paths):

    load_csv = lambda x: RLDatasetFormatter().from_csv(x)   
    dfs = []
    s_max = None
    s_min = None
    a_max = None
    a_min = None
    for path in dataset_paths:
        temp_df = load_csv(path)
        dfs.append(temp_df)
        if s_max is None:
            s_max = torch.max(torch.max(temp_df.s, 0)[0], torch.max(temp_df.s1, 0)[0])
            s_min = torch.min(torch.min(temp_df.s, 0)[0], torch.min(temp_df.s1, 0)[0])
            a_max = torch.max(temp_df.a, 0)[0]
            a_min = torch.min(temp_df.a, 0)[0]
        else:
            s_max = torch.max(torch.max(s_max, torch.max(temp_df.s, 0)[0]), torch.max(temp_df.s1, 0)[0])
            s_min = torch.min(torch.min(s_min, torch.min(temp_df.s, 0)[0]), torch.min(temp_df.s1, 0)[0])
            a_max = torch.max(a_max, torch.max(temp_df.a, 0)[0])
            a_min = torch.min(a_min, torch.max(temp_df.a, 0)[0])

    if prob_name == "MC" or prob_name == "Pend":
        if prob_name == "Pend":
            # Define dataset-specific metadata
            s_max = torch.tensor([1.0, 1.0, 8.0])
            s_min = torch.tensor([-1.0, -1.0, -8.0])
            a_max = torch.tensor([2.0])
            a_min = torch.tensor([-2.0])
        else:
            s_max = torch.tensor([0.6, 0.07])
            s_min = torch.tensor([-1.2, -0.07])
            a_max = torch.tensor([1.0])
            a_min = torch.tensor([-1.0])

    for df in dfs:
        df.normalize_data(s_max, s_min, a_max, a_min)

    s_size = dfs[0].state_size
    a_size = dfs[0].action_size

    return dfs, (s_max, s_min, a_max, a_min), (s_size, a_size)

def problem_factory(source_name, target_name):
    source_csvs_dict = cfg.PROBLEMS_CONFIG[source_name]['csvs']
    target_csvs_dict = cfg.PROBLEMS_CONFIG[target_name]['csvs']

    source_dataset_path = [source_csvs_dict['source'], source_csvs_dict['source_val']]
    target_dataset_path = [target_csvs_dict['target'], target_csvs_dict['target_val'], target_csvs_dict['test']]
    
    source_dfs, source_max_mins, source_size = problem_dispenser(source_name, source_dataset_path)
    target_dfs, target_max_mins, target_size = problem_dispenser(target_name, target_dataset_path)

    source_data = {
        'dfs': source_dfs,
        'max_mins': source_max_mins,
        'sizes': source_size
    }
    target_data = {
        'dfs': target_dfs,
        'max_mins': target_max_mins,
        'sizes': target_size
    }
    return source_data, target_data

            
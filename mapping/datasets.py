import torch

from torch.utils.data import Dataset
from mapping.util import RLCSVReader


class DatasetSkeleton(Dataset):
    def __init__(self, X, Y) -> None:
        super().__init__()
        self.X = X
        self.Y = Y
    
    def __len__(self) -> int:
        return len(self.X) 

    def __getitem__(self, index: int):
        return (self.X[index], self.Y[index])

    def merge(self, dataset):
        self.X = torch.cat([self.X, dataset.X])
        self.Y = torch.cat([self.Y, dataset.Y])
        return self
    
    def shuffle(self):
        rand_index = torch.randperm(self.X.shape[0])
        self.X = self.X[rand_index]
        self.Y = self.Y[rand_index]

    
class RLDatasetFormatter:

    # data = [s,a,r,s']
    # s,a,r,s' are tensors
    def __init__(self, data=None) -> None:
        if data is None:
            self.s = torch.tensor([])
            self.a = torch.tensor([])
            self.r = torch.tensor([])
            self.s1 = torch.tensor([])

            self.state_size = 0
            self.action_size = 0
        else:
            self.set_data(data)
    
    def set_data(self, data):
        self.s = data[0]
        self.a = data[1]
        self.r = data[2]
        self.s1 = data[3]
        self.state_size = self.s.shape[1]
        self.action_size = self.a.shape[1]
    
    def from_csv(self, path):
        self.set_data(RLCSVReader(path).read())
        return self
    
    def as_transitions(self):
        X, Y = (torch.cat([self.s,self.a],1),self.s1)
        return DatasetSkeleton(X,Y)
    
    def _transition_as_label(self, label):
        X = torch.cat([self.s,self.a,self.s1],1)
        Y = torch.zeros((X.shape[0],1)).fill_(label)
        return (X,Y)

    def transition_as_valid(self):
        X, Y = self._transition_as_label(1)
        return DatasetSkeleton(X,Y)

    def transition_as_fake(self):
        X, Y = self._transition_as_label(0)
        return DatasetSkeleton(X,Y)
    
    def transition_identity(self):
        X = torch.cat([self.s,self.a,self.s1],1)
        Y = torch.cat([self.s,self.a,self.s1],1)
        return DatasetSkeleton(X,Y)
    
    def normalize_data(self, max_s, min_s, max_a, min_a):
        norm_s = (self.s - min_s)/(max_s-min_s)
        norm_a = (self.a - min_a)/(max_a-min_a)
        norm_s1 = (self.s1 - min_s)/(max_s-min_s)
        self.s = norm_s
        self.a = norm_a
        self.s1 = norm_s1
        return self
    
    def denormalize_data(self, max_s, min_s, max_a, min_a):
        s = self.s * (max_s - min_s) + min_s
        a = self.a * (max_a - min_a) + min_a
        s1 = self.s1 * (max_s - min_s) + min_s
        self.s = s
        self.a = a
        self.s1 = s1
        return self





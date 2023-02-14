import csv
import torch

class RLCSVReader:
    def __init__(self, csv_path):
        self.csv_path = csv_path

    def read(self, skip_rows: int = 1, skip_columns: int = 1):
        data = [list() for i in range(4)]
        with open(self.csv_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            for i, row in enumerate(csv_reader):
                if i >= skip_rows:
                    for j, c in enumerate(row):
                        if j >= skip_columns:
                            if "[" in c:
                                c_l = list(
                                    filter(lambda ele: ele != "",
                                           c[1:-1].split(" "))
                                )
                                c_array = [float(k) for k in c_l]
                            else:
                                c_array = [float(c)]
                            data[j - skip_columns].append(c_array)
        data = [torch.tensor(d) for d in data]
        
        return data

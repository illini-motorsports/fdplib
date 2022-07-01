from __future__ import annotations
import tqdm
import numpy as np
from fdplib.errors import FileDoesNotExist

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
           'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


class DarabData:


    def __init__(self, filename: str, no_status: bool = False) -> None:
        """
        Initializes the Darabdata class by taking the txt file of data from darab
        and parses it into data, the labels corresponding to the data, and the
        metadata about the data, no status silences the tqdm loading bar
        """


        try:
            num_lines = sum(1 for line in open(filename, 'r', encoding="utf8", errors='ignore'))

        except:
            raise FileDoesNotExist(filename)

        self.filename = filename
        self.data = []
        self.labels = []

        with open(filename, 'r', encoding="utf8", errors='ignore') as FILE:
            if no_status:
                for line in FILE:
                    self.data.append([item.strip().rstrip() for item in line.split("\t")])
            else:
                print("Loading Data ...")
                for line in tqdm.tqdm(FILE, total=num_lines):
                    self.data.append([item.strip().rstrip() for item in line.split("\t")])

        self.metadata = self.data[0:4]
        data_labels_pre = self.data[5]
        self.data = self.data[6:]

        for label in data_labels_pre:
            first_space = label.find(" ")
            first_bracket = label.find("[")
            if first_space == -1:
                self.labels.append(label[0:first_bracket])
            else:
                self.labels.append(label[0:min(first_space, first_bracket)])


    def to_dict(self) -> dict:
        """
        Turns the entire dataset into a dictionary in which the keys are the
        variable names and the values are the datasets corresponding to the
        variable names
        """
        

        ret = {val:[] for val in self.labels}
        lut = {idx:val for idx, val in enumerate(self.labels)}

        for row in self.data:
            for idx, col in enumerate(row):
                ret[lut[idx]].append(col)

        return ret


    def get_var(self, var, timeseries = False) -> list:
        """
        Returns data associated with passed variable. Optionally returns a
        timeseries array of the data if the timeseries optional parameter
        is passed as true
        """
        
        try:
            idx = self.labels.index(var)

        except:
            return None

        if [row[idx] for row in self.data][0][0] in letters:
            if timeseries:
                return [list(map(float, [row[0] for row in self.data])),
                                        [row[idx] for row in self.data]]
            else:
                return [row[idx] for row in self.data]
        else:
            if timeseries:
                return [list(map(float, [row[0] for row in self.data])),
                        list(map(float,[row[idx] for row in self.data]))]
            else:
                return list(map(float,[row[idx] for row in self.data]))


    def get_var_np(self, var, timeseries = False) -> np.ndarray:
        """
        Preforms the same function as get_var except the data is returned 
        in a numpy array instead of a python list
        """
        
        return np.array(self.get_var(var, timeseries))

    
    def list_vars(self) -> list:
        """
        returns a list of the different variables available in the dataset
        """
        
        return self.labels


    # def sub_dataset(self, vars: list) -> DarabData:
        
    #     ret_vars = [var for var in vars if var in self.labels]

    
    def __getitem__(self, key) -> list:
        """
        Accessor function allows for variables to be accessed as though the
        DarabData class was a dictionary, see the to_dict function for
        implementation details
        """

        return self.get_var(key)
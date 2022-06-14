import tqdm
import numpy as np
from fdplib.errors import FileDoesNotExist

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
           'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


class DarabData:


    def __init__(self, filename: str) -> None:
        """
        Initializes the Darabdata class by taking the txt file of data from darab
        and parses it into data, the labels corresponding to the data, and the
        metadata about the data
        """


        try:
            num_lines = sum(1 for line in open(filename,'r'))

        except:
            raise FileDoesNotExist(filename)

        self.filename = filename
        self.data = []
        self.labels = []

        print("Loading Data ...")
        with open(filename,'r') as FILE:
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
        
        idx = self.labels.index(var)

        if idx == -1:
            return None

        else:
            if self.data[idx][0][0] in letters:
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


    def get_var_np(self, var, timeseries = False):
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

    
    def __getitem__(self, key) -> list:
        """
        Accessor function allows for variables to be accessed as though the
        DarabData class was a dictionary, see the to_dict function
        """

        idx = self.labels.index(key)
        if idx == -1:
            return None
        else:
            return [row[idx] for row in self.data]
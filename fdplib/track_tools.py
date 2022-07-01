import numpy as np
from pyrsistent import s
from fdplib.errors import FileDoesNotExist
from fdplib.darab import DarabData


class Track:
    
    def __init__(self):
       pass

    def from_data(self, filepath: str) -> None:
        """import darab text file"""
        
        self._data = DarabData(filepath, no_status=True)

t = Track()

t.from_data("/Users/collin/code/fdplib/test_data/May_MIS_END_MATT_Track.txt")
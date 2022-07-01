from tempfile import TemporaryFile
import numpy as np
from fdplib.errors import FileDoesNotExist

class Track:
    
    def __init__(self):
       pass

    def from_data(self, filepath: str) -> None:
        try:
            with open(filepath, "r") as FILE:
                self._data_file = FILE.read()

        except FileNotFoundError as e:
            raise FileDoesNotExist(filepath) from None

t = Track()

t.from_data("test")
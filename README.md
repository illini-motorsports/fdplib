# Formula Data Processing Library
## _The python windarab loader_

UNIT TESTS WIP

Formula data processing library or fdplib for short is a library for accessing the custom text file format exported by Windarab by Bosch Motorsports

## Installation

FDPLib only requires tqdm (used as status bar for loading large data files)

To install FDPLib use pip:

```sh
python3 -m pip install fdplib
```

## Usage

In order to use the main functionality of this library, import the fdplib module in your python code
```
import fdplib
```
From this a *DarabData* object can be created by passing the textfile name to the constructor
```
import fdplib.darab as darab
data = darab.DarabData("<FILENAME>.txt")
```
With this object, data can be easily accessed using the following methods. First, to get all of the variables in the dataset do ```data.list_vars()```

The main method that you will use to interact with the data is ```data.get_var("<VARIABLE_NAME>")```
This method returns a list of all the datapoints associated with the variable. Likewise ```data.get_var_np("<VARIABLE_NAME>")``` returns a numpy array of the data.
There is an optional boolean parameter called timeseries for both of the previously mentioned methods. If set to True this instead returns the data and the time values associated with it as a 2D array, much in the way matlab does.
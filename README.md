# Formula Data Processing Library
## _The python windarab loader_

[![Unit Tests](https://github.com/illini-motorsports/fdplib/actions/workflows/main_unit_tests.yml/badge.svg)](https://github.com/illini-motorsports/fdplib/actions/workflows/main_unit_tests.yml)
![Coverage Report](https://raw.githubusercontent.com/illini-motorsports/fdplib/main/coverage.svg)

Formula data processing library or fdplib for short is a library for accessing the custom text file format exported by Windarab by Bosch Motorsports

## Installation

FDPLib requires tqdm (used as status bar for loading large data files) and Numpy (used to return data in numpy arrays)

To install FDPLib use pip:

```sh
python3 -m pip install fdplib
```

## Usage
#### Common Usage
In order to use the main module of this library, import the fdplib module in your python code
```python
import fdplib
```
A *DarabData* object can be created by passing the textfile name to the constructor
```python
import fdplib.darab as darab
data = darab.DarabData("<FILENAME>.txt")
```
With this object, data can be easily accessed using the following methods. First, to get all of the variables in the dataset do ```data.list_vars()```

The main method that you will use to interact with the data is ```data.get_var("<VARIABLE_NAME>")```
This method returns a list of all the datapoints associated with the variable. Likewise ```data.get_var_np("<VARIABLE_NAME>")``` returns a numpy array of the data.
There is an optional boolean parameter called timeseries for both of the previously mentioned methods. If set to True this instead returns the data and the time values associated with it as a 2D array, much in the way matlab does.

#### More Advanced Usage
fdplib also has a feature rich ```class Track``` which can be used to analyze and breakdown lap data from the vehicle. To access this class do:
```python
import fdplib.track_tools as tt
track = tt.Track("<FILENAME>.txt")
```
Using this class the following methods are available:
```python
def plot_track_heatmap(self, t_bound: tuple = None, direct_arrow: bool = None,
                           heat_source: np.array = None) -> None:
        """plot a heatmap of speed from gps data in matplotlib"""
```
```python
def coords_from_gps(self) -> np.array:
    """converts longitude and latitude data to x and y displacements
       coordinates from the first location coordinate"""
```
```python
def coords_from_acc(self, ret_yaw: bool = False) -> np.array:
    """calculates the vehicles path from acceleration data provided by the IMU.
       If ret_yaw is true also return the corresponding yaw data."""
```
```python
def plot_coords(self, coords: np.array) -> None:
    """plots x and y displacement coordinates"""
```
```python
def radius_from_gps_coords(self, coords: np.array) -> np.array:
    """Calculates the radius of the current turn that the vehicle is in from gps data.
       This is done using a 3 points on a circle calculation and iterates through all the data passed"""
```
```python
def get_lap_bounds(self, lap_num: int) -> np.array:
    """Calculate the indexs in the data corresponding to the (start,end) of
       the requested lap given by lap_num."""
```
```python
def simulate(self) -> None: # pragma: no cover
    """Simulate lap data using pygame"""
```

## Examples

![example nmot data](https://raw.githubusercontent.com/illini-motorsports/fdplib/main/assets/example_nmot.png)

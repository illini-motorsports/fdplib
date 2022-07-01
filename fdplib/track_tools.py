import numpy as np
import matplotlib.pyplot as plt
from fdplib.errors import VariableNotPresent
from fdplib.darab import DarabData


class Track:
    
    def __init__(self):
       pass

    def from_data(self, filepath: str) -> None:
        """import darab text file and store darabdata in obj"""
        
        self._data = DarabData(filepath, no_status=True)

    def plot_track(self, t_bound: tuple = None) -> None:
        """plot a map from gps data in matplotlib"""
        # ms6 vars : GPS_Lat, GPS_Long
        lat = self._data.get_var("GPS_Lat")
        long = self._data.get_var("GPS_Long")

        if not lat:
            raise VariableNotPresent("GPS_Lat")
        if not long:
            raise VariableNotPresent("GPS_Long")

        if not t_bound is None:
            xtime = self._data.get_var("xtime")
            if not xtime:
                raise VariableNotPresent("xtime")
            xtime = np.array(xtime)
            start_idx = np.argmin(np.abs(xtime-t_bound[0]))
            end_idx = np.argmin(np.abs(xtime-t_bound[1]))

            lat = lat[start_idx:end_idx]
            long = long[start_idx:end_idx]

        plt.plot(long, lat)
        plt.show()

    def plot_track_heatmap(self, t_bound: tuple = None) -> None:
        """plot a heatmap of speed from gps data in matplotlib"""
        lat = self._data.get_var("GPS_Lat")
        long = self._data.get_var("GPS_Long")
        speed = self._data.get_var("speed")

        if not lat:
            raise VariableNotPresent("GPS_Lat")
        if not long:
            raise VariableNotPresent("GPS_Long")
        if not speed:
            raise VariableNotPresent("speed")

        if not t_bound is None:
            xtime = self._data.get_var("xtime")
            if not xtime:
                raise VariableNotPresent("xtime")
            xtime = np.array(xtime)
            start_idx = np.argmin(np.abs(xtime-t_bound[0]))
            end_idx = np.argmin(np.abs(xtime-t_bound[1]))

            lat = lat[start_idx:end_idx]
            long = long[start_idx:end_idx]
            speed = speed[start_idx:end_idx]
            
        plt.scatter(long, lat, c=speed, cmap='viridis')
        plt.show()

from turtle import clear
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from fdplib.errors import VariableNotPresent
from fdplib.darab import DarabData
from fdplib.track_classes import GPSCoord
from math import sin, cos, radians


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

        plt.plot(lat, long)

    def plot_track_heatmap(self, t_bound: tuple = None, direct_arrow: bool = None) -> None:
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

        lat = np.array(lat)
        long = np.array(long)

        if not (t_bound is None):
            xtime = self._data.get_var("xtime")
            if not xtime:
                raise VariableNotPresent("xtime")
            xtime = np.array(xtime)
            start_idx = np.argmin(np.abs(xtime-t_bound[0]))
            end_idx = np.argmin(np.abs(xtime-t_bound[1]))

            lat = lat[start_idx:end_idx]
            long = long[start_idx:end_idx]
            speed = speed[start_idx:end_idx]
        
        if direct_arrow:
            center_x = (1/len(long))*np.sum(long)
            center_y = (1/len(lat))*np.sum(lat)
            delta_x = np.max(long)-np.min(long)

            direction = self._calc_track_direction(long, lat, (center_x, center_y))

            if direction:
                sign = ''
            else:
                sign = '-'

            start = GPSCoord((center_x-(delta_x/15), center_y))
            end = GPSCoord((center_x+(delta_x/15), center_y))

            curl_arrow = patches.FancyArrowPatch(start.lat_long(), end.lat_long(), connectionstyle=f"arc3,rad={sign}.3",
                                                **{"arrowstyle":"Simple, tail_width=0.5, head_width=4, head_length=8"})
            plt.gca().add_patch(curl_arrow)

        plt.scatter(long, lat, c=speed, cmap='magma')
        plt.colorbar()

    def _calc_track_direction(self, long, lat, center):
        """uses lat and long data to determine if direction of travel is clockwise
           or counter-clockwise. This is done by analyzing the directive of the change
           of angle from a static point in the middle of the track."""
        vector_x = long-center[0]
        vector_y = lat-center[1]

        angle = np.arctan(vector_y/vector_x)
        mean = np.mean(np.gradient(angle, 0.00001))
        
        return (mean > 0)

    def coords_from_gps(self) -> np.array:
        """converts longitude and latitude data to x and y displacements
           coordinates from the first location coordinate"""
        lat = self._data.get_var("GPS_Lat")
        long = self._data.get_var("GPS_Long")
        speed = self._data.get_var("speed")

        if not lat:
            raise VariableNotPresent("GPS_Lat")
        if not long:
            raise VariableNotPresent("GPS_Long")
        if not speed:
            raise VariableNotPresent("speed")

        lat = np.array(lat)
        long = np.array(long)

        start_coord = GPSCoord((lat[0], long[0]))
        coords = [GPSCoord((lati, longi)) for lati, longi in zip(lat, long)][1:]

        distances = []

        for coord in coords:
            distances.append(start_coord.distance(coord))

        x_vals = np.array([item[0] for item in distances])*1000
        y_vals = np.array([item[1] for item in distances])*1000

        return np.vstack((x_vals, y_vals))

    def plot_coords(self, coords: np.array) -> None:
        """plots x and y displacement coordinates"""
        plt.plot(coords[0], coords[1], label="Position")
        plt.plot(0, 0, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="black", label="Start/Origin")
        plt.xlabel("x distance (m)")
        plt.ylabel("y distance (m)")
        plt.title("Track map, distance from origin (m)")
        plt.legend()

    def coords_from_acc(self):
        acc_lat = self._data.get_var("accy")
        acc_long = self._data.get_var("accx")
        speed = self._data.get_var("speed")
        yaw_rate = self._data.get_var("yaw")

        if not acc_lat:
            raise VariableNotPresent("accy")
        if not acc_long:
            raise VariableNotPresent("accx")
        if not speed:
            raise VariableNotPresent("speed")
        if not yaw_rate:
            raise VariableNotPresent("yaw")

        vel_y = np.zeros(len(acc_long))
        pos_y = np.zeros(len(acc_long))
        vel_x = np.zeros(len(acc_long))
        pos_x = np.zeros(len(acc_long))
        yaw = np.zeros(len(acc_long))

        yaw[0] = 240
        vel_x[0] = speed[0]*sin(radians(yaw[0]))
        vel_y[0] = speed[0]*cos(radians(yaw[0]))

        dt = 0.01
        for i in range(len(acc_lat)-1):
            yaw[i+1] = yaw[i] + yaw_rate[i]*dt
            vel_y[i] = speed[i] * cos(radians(yaw[i]))
            vel_x[i] = speed[i] * sin(radians(yaw[i]))
            pos_x[i+1] = pos_x[i] + vel_x[i]*dt
            pos_y[i+1] = pos_y[i] + vel_y[i]*dt
        
        return np.vstack((pos_x, pos_y))

    def radius_from_gps_coords(self, coords: np.array) -> np.array:
        x_c = coords[0]
        y_c = coords[1]
        coords = np.vstack((np.array(x_c[::5]),np.array(y_c[::5])))
        c = len(coords[0])
        p = coords.T

        c_dist = np.zeros(c)
        b_dist = np.zeros(c)
        a_dist = np.zeros(c)
        A_angle = np.zeros(c)
        R = np.zeros(c)

        for i in range(c-2):
            c_dist[i] = ((p[i+1,0]-p[i,0])**2+(p[i+1,1]-p[i,1])**2)**(0.5)
            b_dist[i] = ((p[i+2,0]-p[i+1,0])**2+(p[i+2,1]-p[i+1,1])**2)**(0.5)
            a_dist[i] = ((p[i+2,0]-p[i,0])**2+(p[i+2,1]-p[i,1])**2)**(0.5)
            A_angle[i] = np.arccos(radians(b_dist[i]**2 + c_dist[i]**2-a_dist[i]**2))/(2*b_dist[i]*c_dist[i])
            R[i] = a_dist[i]/2*np.sin(radians(180-A_angle[i]))

        R = R*1000 # kilometers to meters
        
        out = []
        for rad in R:
            for i in range(5):
                out.append(rad)

        return np.array(out)
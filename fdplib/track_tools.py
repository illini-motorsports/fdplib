import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from fdplib.errors import VariableNotPresent
from fdplib.darab import DarabData
from fdplib.track_classes import GPSCoord, Vector
from math import sin, cos, radians



# =============================================================================
# =============================================================================



class Track:
    """Class representing a track on which our formula vehicle travels.
       It is initialized by a data file with several necessary data points.
       It allows the user to calculate and display many intriguing aspects of the data."""

    def __init__(self, filepath: str = None) -> None:
        """If a path is provided, initializes the data from file path"""
        if filepath:
            self._data = DarabData(filepath, no_status=True)

    # -------------------------------------------------------------------------

    def from_data(self, filepath: str) -> None:
        """import darab text file and store darabdata in obj"""
        self._data = DarabData(filepath, no_status=True)

    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------

    def plot_track_heatmap(self, t_bound: tuple = None, direct_arrow: bool = None,
                           heat_source: np.array = None) -> None:
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

        if heat_source:
            c = heat_source
        else:
            c = speed

        plt.scatter(long, lat, c=c, cmap='magma')
        plt.colorbar()
    
    # -------------------------------------------------------------------------

    def _calc_track_direction(self, long, lat, center) -> bool:
        """uses lat and long data to determine if direction of travel is clockwise
           or counter-clockwise. This is done by analyzing the directive of the change
           of angle from a static point in the middle of the track."""
        vector_x = long-center[0]
        vector_y = lat-center[1]

        angle = np.arctan(vector_y/vector_x)
        mean = np.mean(np.gradient(angle, 0.00001))
        
        return (mean > 0)
    
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------

    def plot_coords(self, coords: np.array) -> None:
        """plots x and y displacement coordinates"""
        plt.plot(coords[0], coords[1], label="Position")
        plt.plot(0, 0, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="black", label="Start/Origin")
        plt.xlabel("x distance (m)")
        plt.ylabel("y distance (m)")
        plt.title("Track map, distance from origin (m)")
        plt.legend()

    def coords_from_acc(self, ret_yaw: bool = False) -> np.array:
        """calculates the vehicles path from acceleration data provided by the IMU.
           If ret_yaw is true also return the corresponding yaw data."""
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

        yaw[0] = 180
        vel_x[0] = speed[0]*sin(radians(yaw[0]))
        vel_y[0] = speed[0]*cos(radians(yaw[0]))

        dt = 0.01
        for i in range(len(acc_lat)-1):
            yaw[i+1] = yaw[i] + yaw_rate[i]*dt
            vel_y[i] = speed[i] * cos(radians(yaw[i]))
            vel_x[i] = speed[i] * sin(radians(yaw[i]))
            pos_x[i+1] = pos_x[i] + vel_x[i]*dt
            pos_y[i+1] = pos_y[i] + vel_y[i]*dt
        
        if ret_yaw:
            return np.vstack((pos_x, pos_y)), yaw
        else:
            return np.vstack((pos_x, pos_y))
    
    # -------------------------------------------------------------------------

    def radius_from_gps_coords(self, coords: np.array) -> np.array:
        """Calculates the radius of the current turn that the vehicle is in from gps data.
           This is done using a 3 points on a circle calculation and iterates through all the data passed"""
        x_c = coords[0]
        y_c = coords[1]
        coords = np.vstack((np.array(x_c[::5]),np.array(y_c[::5])))
        rads = []

        for idx in range(2, len(coords[0])):
            p1 = coords[:,idx-2]
            p2 = coords[:,idx-1]
            p3 = coords[:,idx]

            x1, y1 = p1[0], p1[1]
            x2, y2 = p2[0], p2[1]
            x3, y3 = p3[0], p3[1]

            A = x1*(y2-y3) - y1*(x2-x3) + x2*y3 - x3*y2
            B = (x1**2 + y1**2)*(y3-y2) + (x2**2 + y2**2)*(y1-y3) + (x3**2+y3**2)*(y2-y1)
            C = (x1**2 + y1**2)*(x2-x3) + (x2**2 + y2**2)*(x3-x1) + (x3**2+y3**2)*(x1-x2)
            D = (x1**2 + y1**2)*(x3*y2 - x2*y3) + (x2**2 + y2**2)*(x1*y3 - x3*y1) + (x3**2+y3**2)*(x2*y1-x1*y2)

            if A != 0:
                R = np.sqrt((B**2 + C**2 - 4*A*D)/(4*(A**2)))
            else:
                R = 5000
            
            rads.append(R)

        out = [0] * 10
        for r in rads:
            for i in range(5):
                out.append(r)

        # x_c = coords[0]       #RANDOM ALGORITHM THAT KINDA LOOKS RIGHT BUT NOT SURE WHY
        # y_c = coords[1]
        # coords = np.vstack((np.array(x_c[::5]),np.array(y_c[::5])))
        # c = len(coords[0])
        # p = coords.T

        # c_dist = np.zeros(c)
        # b_dist = np.zeros(c)
        # a_dist = np.zeros(c)
        # A_angle = np.zeros(c)
        # R = np.zeros(c)

        # for i in range(c-2):
        #     c_dist[i] = ((p[i+1,0]-p[i,0])**2+(p[i+1,1]-p[i,1])**2)**(0.5)
        #     b_dist[i] = ((p[i+2,0]-p[i+1,0])**2+(p[i+2,1]-p[i+1,1])**2)**(0.5)
        #     a_dist[i] = ((p[i+2,0]-p[i,0])**2+(p[i+2,1]-p[i,1])**2)**(0.5)
        #     A_angle[i] = np.arccos(radians(b_dist[i]**2 + c_dist[i]**2-a_dist[i]**2))/(2*b_dist[i]*c_dist[i])
        #     R[i] = a_dist[i]/(2*np.sin(radians(180-A_angle[i])))

        # out = []
        # for rad in R:
        #     for i in range(5):
        #         out.append(rad)

        return np.array(out)

    # -------------------------------------------------------------------------
    
    def get_lap_bounds(self, lap_num: int) -> np.array:
        """Calculate the indexs in the data corresponding to the (start,end) of
           the requested lap given by lap_num."""
        lat = self._data.get_var("GPS_Lat")
        long = self._data.get_var("GPS_Long")
        xtime = self._data.get_var("xtime")

        if not lat:
            raise VariableNotPresent("GPS_Lat")
        if not long:
            raise VariableNotPresent("GPS_Long")
        if not xtime:
            raise VariableNotPresent("xtime")

        coords = self.coords_from_gps()
        coords_x = coords[0]
        coords_y = coords[1]
        s_x = coords_x[0]
        s_y = coords_y[0]

        curr_lap = 0
        in_rad = True
        idx = 0
        lap_start_idx = 0

        for x, y in zip(coords_x[1:], coords_y[1:]): #iterate through both x and y at same time
            if (self._coords_dist(s_x, x, s_y, y) < 12):
                if not in_rad:
                    curr_lap +=1
                    lap_start_idx = idx
                    in_rad = True
            else:
                if in_rad:
                    in_rad = False
            
            if curr_lap == lap_num:
                return (lap_start_idx, idx)

            idx += 1
    
        return None
    
    # -------------------------------------------------------------------------

    def _coords_dist(self, x1: float, x2: float, y1: float, y2: float) -> float:
        """Returns the distance betweens two x and y coordinates."""
        return np.sqrt((x2-x1)**2 + (y2-y1)**2)

    # -------------------------------------------------------------------------



# =============================================================================
# =============================================================================
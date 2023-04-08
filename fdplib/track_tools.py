import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from fdplib.errors import VariableNotPresent
from fdplib.darab import DarabData
from fdplib.track_classes import GPSCoord
from math import sin, cos, radians


# =============================================================================
# =============================================================================



class Track:
    """Class representing a track on which our formula vehicle travels.
       It is initialized by a data file with several necessary data points.
       It allows the user to calculate and display many intriguing aspects of the data."""

    # -------------------------------------------------------------------------

    def __init__(self, filepath: str) -> None:
        """If a path is provided, initializes the data from file path"""
        self._data = DarabData(filepath, no_status=True)
        
        e = self._data_errors()
        if e:
            raise VariableNotPresent(str(e))

    # -------------------------------------------------------------------------

    # def plot_track(self, t_bound: tuple = None) -> None:      |   DEPRECATED! DATA
    #     """plot a map from gps data in matplotlib"""          |   SHOULD BE OBTAINED
    #     # ms6 vars : GPS_Lat, GPS_Long                        |   FROM GPS OR ACC THEN
    #     lat = self._data.get_var_np("GPS_Lat")                |   PLOTTED USING:
    #     long = self._data.get_var_np("GPS_Long")              |   PLOT COORDS
    #                                                           |
    #     if not t_bound is None:                               |
    #         xtime = self._data.get_var_np("xtime")            |
    #         start_idx = np.argmin(np.abs(xtime-t_bound[0]))   |
    #         end_idx = np.argmin(np.abs(xtime-t_bound[1]))     |
    #                                                           |
    #         lat = lat[start_idx:end_idx]                      |
    #         long = long[start_idx:end_idx]                    |
    #                                                           |
    #     plt.plot(lat, long)                                   |

    # -------------------------------------------------------------------------

    def plot_track_heatmap(self, t_bound: tuple = None, direct_arrow: bool = None,
                           heat_source: np.array = None) -> None:
        """plot a heatmap of speed from gps data in matplotlib"""
        lat = self._data.get_var_np("GPS_Lat")
        long = self._data.get_var_np("GPS_Long")
        speed = self._data.get_var("speed")

        if not (t_bound is None):
            xtime = self._data.get_var_np("xtime")
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

        if type(heat_source) != None:
            c = heat_source
        else:
            c = speed

        plt.scatter(long, lat, c=c, cmap='magma')
        plt.colorbar()
    
    # -------------------------------------------------------------------------

    def coords_from_gps(self) -> np.array:
        """converts longitude and latitude data to x and y displacements
           coordinates from the first location coordinate"""
        lat = self._data.get_var_np("GPS_Lat")
        long = self._data.get_var_np("GPS_Long")

        start_coord = GPSCoord((lat[0], long[0]))
        coords = [GPSCoord((lati, longi)) for lati, longi in zip(lat, long)][1:]

        distances = []

        for coord in coords:
            distances.append(start_coord.distance(coord))

        x_vals = np.array([item[0] for item in distances])*1000
        y_vals = np.array([item[1] for item in distances])*1000

        return np.vstack((x_vals, y_vals))

    # -------------------------------------------------------------------------

    def coords_from_acc(self, ret_yaw: bool = False) -> np.array:
        """calculates the vehicles path from acceleration data provided by the IMU.
           If ret_yaw is true also return the corresponding yaw data."""
        acc_lat = self._data.get_var_np("accy")
        acc_long = self._data.get_var_np("accx")
        speed = self._data.get_var_np("speed")
        yaw_rate = self._data.get_var_np("yaw")

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
        
        if ret_yaw:
            return np.vstack((pos_x, pos_y)), yaw
        else:
            return np.vstack((pos_x, pos_y))

    # -------------------------------------------------------------------------

    def plot_coords(self, coords: np.array) -> None:
        """plots x and y displacement coordinates"""
        plt.plot(coords[0], coords[1], label="Position")
        plt.plot(coords[0][0], coords[1][0], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="black", label="Start/Origin")
        plt.xlabel("x distance (m)")
        plt.ylabel("y distance (m)")
        plt.title("Track map, distance from origin (m)")
        plt.legend()
    
    # -------------------------------------------------------------------------

    def radius_from_gps_coords(self, coords: np.array) -> np.array:
        """Calculates the radius of the current turn that the vehicle is in from gps data.
           This is done using a 3 points on a circle calculation and iterates through all the data passed"""
        x_c = coords[0]
        y_c = coords[1]
        coords = np.vstack((np.array(x_c[::5]),np.array(y_c[::5])))
        rads = []

        for idx in range(1, len(coords[0])-1):
            
            x1, y1 = coords[0][idx-1], coords[1][idx-1]
            x2, y2 = coords[0][idx], coords[1][idx]
            x3, y3 = coords[0][idx+1], coords[1][idx+1]
            c = (x1-x2)**2 + (y1-y2)**2
            a = (x2-x3)**2 + (y2-y3)**2
            b = (x3-x1)**2 + (y3-y1)**2
            ar = a**0.5
            br = b**0.5
            cr = c**0.5 
            r = ar*br*cr / ((ar+br+cr)*(-ar+br+cr)*(ar-br+cr)*(ar+br-cr))**0.5
            rads.append(r)

        rads.append(300)
        rads.append(300)

        rads = np.clip(rads, 5, 100)

        plt.scatter(coords[0], coords[1], c=rads, cmap='magma')

        out = [300] * 7
        for r in rads:
            for i in range(5):
                out.append(r)

        out = np.clip(out, 5, 300)
        
        return np.array(out)

    # -------------------------------------------------------------------------
    
    def get_lap_bounds(self, lap_num: int) -> np.array:
        """Calculate the indexs in the data corresponding to the (start,end) of
           the requested lap given by lap_num."""

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
                    in_rad = True
            else:
                if in_rad:
                    lap_start_idx = idx
                    in_rad = False
            
            if curr_lap == lap_num:
                return (lap_start_idx, idx)

            idx += 1
    
        return None
    
    # -------------------------------------------------------------------------


    def simulate(self) -> None: # pragma: no cover
        """Simulate lap data using pygame"""
        WIDTH = 1680
        HEIGHT = 1000
        FPS = 60
        STEP = 5
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        RED = (255, 0, 0)
        GREEN = (0, 255, 0)
        BLUE = (0, 0, 255)

        import pygame

        pygame.init()
        pygame.mixer.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("<Your game>")
        clock = pygame.time.Clock()

        carImg = pygame.image.load("/Users/collin/code/fdplib/assets/car_sprite.png")

        def draw_img(image, x, y, angle):
            rotated_image = pygame.transform.rotate(image, angle) 
            screen.blit(rotated_image, rotated_image.get_rect(center=image.get_rect(topleft=(x, y)).center).topleft)


        coords, yaw = self.coords_from_acc(ret_yaw=True)
        coords[0] += -1*np.min(coords[0]) # | bring all values into positive area
        coords[1] += -1*np.min(coords[1]) # |

        coords[0] *= (WIDTH * 0.9)/np.max(coords[0])
        coords[1] *= (HEIGHT * 0.9)/np.max(coords[1])

        idx = 0
        max_idx = len(coords[0])
        
        running = True
        while running:

            clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False


            screen.fill(WHITE)

            draw_img(carImg, int(coords[0][idx]), int(coords[1][idx]), yaw[idx])

            if idx+STEP >= max_idx:
                idx = 0
            else:
                idx += STEP

            pygame.display.flip()       

        pygame.quit()

    # -------------------------------------------------------------------------

    def _coords_dist(self, x1: float, x2: float, y1: float, y2: float) -> float:
        """Returns the distance betweens two x and y coordinates."""
        return np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
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

    def _data_errors(self) -> list:
        """Goes through the data in the dataset and ensures all necessary variabls
           are present."""
        ret = []
        if not self._data.get_var("xtime"):
            ret.append("xtime")
        if not self._data.get_var("GPS_Long"):
            ret.append("GPS_Long")
        if not self._data.get_var("GPS_Lat"):
            ret.append("GPS_Lat")
        if not self._data.get_var("speed"):
            ret.append("speed")
        if not self._data.get_var("accx"):
            ret.append("accx")
        if not self._data.get_var("accy"):
            ret.append("accy")
        if not self._data.get_var("yaw"):
            ret.append("yaw")

        if len(ret) == 0:
            return None
        else:
            return ret

    # -------------------------------------------------------------------------

# =============================================================================
# =============================================================================
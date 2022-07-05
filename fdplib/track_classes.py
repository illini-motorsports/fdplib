from __future__ import annotations
import matplotlib.pyplot as plt
from math import radians, tan
import numpy as np



# =============================================================================
# =============================================================================



class GPSCoord: # pragma: no cover
    """A class to store and manipulate latitude and longitude data"""

    # -------------------------------------------------------------------------

    def __init__(self, init_coords: tuple) -> None:
        """initializes the coords with latitude and longitude by a tuple
           as (longitude: float, latitude: float)"""
        self.long = init_coords[0]
        self.lat = init_coords[1]

    # -------------------------------------------------------------------------

    def plot(self) -> None:
        """plots point via matplotlib"""
        plt.plot([self.long], [self.lat], marker="o", markersize=10, markeredgecolor="white", markerfacecolor="red")

    # -------------------------------------------------------------------------

    def abs_dy(self, other: GPSCoord) -> float:
        """return absolute difference in latitude between coords"""
        return abs(self.lat-other.lat)

    # -------------------------------------------------------------------------

    def abs_dx(self, other: GPSCoord) -> float:
        """return absolute difference in longitude between coords"""
        return abs(self.long-other.long)

    # -------------------------------------------------------------------------

    def dy(self, other: GPSCoord) -> float:
        """return difference in latitude between coords"""
        return self.lat-other.lat

    # -------------------------------------------------------------------------

    def dx(self, other: GPSCoord) -> float:
        """return difference in longitude between coords"""
        return self.long-other.long

    # -------------------------------------------------------------------------

    def lat_long(self) -> tuple:
        """return latitude and longitude as tuple"""
        return (self.long,self.lat)

    # -------------------------------------------------------------------------
    
    def distance(self, other: GPSCoord) -> tuple:
        """calculate the distance between two lat/long coords
           by interpreting them as flat plane x and y coords and
           calculating distance"""
        lon1 = radians(self.long)
        lon2 = radians(other.long)
        lat1 = radians(self.lat)
        lat2 = radians(other.lat)
        
        d_lon = lon2 - lon1
        d_lat = lat2 - lat1

        x_dist = tan(d_lon)*6371
        y_dist = tan(d_lat)*6371

        return (x_dist, y_dist)

    # -------------------------------------------------------------------------



# =============================================================================
# =============================================================================



class Vector: # pragma: no cover
    """A class representing a line segment or vector between two points
       start and end, storing it as x and y representing deltas from the origin
       on the x and y axis"""

    # -------------------------------------------------------------------------

    def __init__(self, start: np.array, end: np.array) -> None:
        """calculate x and y delta from start and end (2,) np arrays"""
        self.start = start
        self.end = end

        self.x = end[0]-start[0]
        self.y = end[1]-start[1]

    # -------------------------------------------------------------------------

    def len_sq(self) -> float:
        """calculate the squared length of the vector"""
        return (self.x**2) + (self.y**2)

    # -------------------------------------------------------------------------

    def len(self) -> float:
        """calculate the length of the vector"""
        return np.sqrt((self.x**2) + (self.y**2))

    # -------------------------------------------------------------------------

    def dot(self, other: Vector) -> float:
        """calculate the dot product between this vector and another"""
        return np.dot([self.x, self.y], [other.x, other.y])

    # -------------------------------------------------------------------------

    def cross(self, other: Vector) -> np.array:
        """calculate the cross product between this vector and another"""
        return np.cross([self.x, self.y], [other.x, other.y])

    # -------------------------------------------------------------------------



# =============================================================================
# =============================================================================
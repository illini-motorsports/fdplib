from __future__ import annotations
import matplotlib.pyplot as plt
from math import radians, tan
import numpy as np

class GPSCoord:
    def __init__(self, init_coords: tuple) -> None:
        """initializes the coords with latitude and longitude by a tuple
           as (longitude: float, latitude: float)"""
        self.long = init_coords[0]
        self.lat = init_coords[1]

    def plot(self):
        """plots point via matplotlib"""
        plt.plot([self.long], [self.lat], marker="o", markersize=10, markeredgecolor="white", markerfacecolor="red")

    def abs_dy(self, other: GPSCoord):
        """return absolute difference in latitude between coords"""
        return abs(self.lat-other.lat)

    def abs_dx(self, other: GPSCoord):
        """return absolute difference in longitude between coords"""
        return abs(self.long-other.long)

    def dy(self, other: GPSCoord):
        """return difference in latitude between coords"""
        return self.lat-other.lat

    def dx(self, other: GPSCoord):
        """return difference in longitude between coords"""
        return self.long-other.long

    def lat_long(self):
        return (self.long,self.lat)
    
    def distance(self, other: GPSCoord) -> tuple:
        lon1 = radians(self.long)
        lon2 = radians(other.long)
        lat1 = radians(self.lat)
        lat2 = radians(other.lat)
        
        d_lon = lon2 - lon1
        d_lat = lat2 - lat1

        x_dist = tan(d_lon)*6371
        y_dist = tan(d_lat)*6371

        return (x_dist, y_dist)

class Vector:
    
    def __init__(self, start: np.array, end: np.array) -> None:
        self.start = start
        self.end = end

        self.x = end[0]-start[0]
        self.y = end[1]-start[1]

    def len_sq(self) -> float:
        return (self.x**2) + (self.y**2)

    def len(self) -> float:
        return np.sqrt((self.x**2) + (self.y**2))

    def dot(self, other: Vector):
        return np.dot([self.x, self.y], [other.x, other.y])

    def cross(self, other: Vector):
        return np.cross([self.x, self.y], [other.x, other.y])
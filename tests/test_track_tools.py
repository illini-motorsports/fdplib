import pytest

from fdplib.track_tools import Track
from fdplib.errors import VariableNotPresent


def test_Track_initialization(track_data_path):
    t = Track(track_data_path)

    assert t._data.get_var("xtime") != None


def test_Track_from_data(track_data_path):
    t = Track()
    t.from_data(track_data_path)

    assert t._data.get_var("xtime") != None


def test_plot_track_full(track_data_path):
    t = Track(track_data_path)

    t.plot_track()


def test_plot_track_bounded(track_data_path):
    t = Track(track_data_path)

    t.plot_track(t_bound=(0, 100))


def test_plot_track_missing_lat(track_data_path):
    t = Track(track_data_path)

    t._data.labels.remove("GPS_Long")

    with pytest.raises(VariableNotPresent) as e_info:
        t.plot_track()


def test_plot_track_missing_long(track_data_path):
    t = Track(track_data_path)

    t._data.labels.remove("GPS_Lat")

    with pytest.raises(VariableNotPresent) as e_info:
        t.plot_track()

def test_plot_track_missing_long(track_data_path):
    t = Track(track_data_path)

    t._data.labels.remove("xtime")

    with pytest.raises(VariableNotPresent) as e_info:
        t.plot_track(t_bound=(0,100))
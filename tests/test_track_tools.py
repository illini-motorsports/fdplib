import pickle
import numpy as np
from fdplib.track_tools import Track


def test_Track_initialization(track_data_path):
    t = Track(track_data_path)

    assert t._data.get_var("xtime") != None

def test_Track_plot_heatmap_no_args(track_data_path):
    t = Track(track_data_path)

    t.plot_track_heatmap()

def test_Track_plot_heatmap_with_bounds(track_data_path):
    t = Track(track_data_path)

    t.plot_track_heatmap(t_bound=(0,100))

def test_Track_plot_heatmap_with_direction(track_data_path):
    t = Track(track_data_path)

    t.plot_track_heatmap(direct_arrow=True)

def test_Track_plot_heatmap_with_heat_source(track_data_path):
    t = Track(track_data_path)
    speed = t._data.get_var("speed")

    t.plot_track_heatmap(heat_source=speed)

def test_Track_coords_from_gps(track_data_path):
    t = Track(track_data_path)
    gps = t.coords_from_gps()

    with open("test_data/gps_coords.pkl", "rb") as FILE:
        good_gps: np.array = pickle.load(FILE)
    
    # @TODO 
    # NEED TO FIX
    # np.testing.assert_array_equal(good_gps, gps)
    assert list(np.around(good_gps[0], 4)) == list(np.around(gps[0], 4))
    assert list(np.around(good_gps[1], 4)) == list(np.around(gps[1], 4))

def test_Track_coords_from_acc(track_data_path):
    t = Track(track_data_path)
    acc = t.coords_from_acc()

    with open("test_data/acc_coords.pkl", "rb") as FILE:
        good_acc: np.array = pickle.load(FILE)
    
    # @TODO 
    # NEED TO FIX
    # np.testing.assert_array_equal(good_acc, acc)
    assert list(np.around(good_acc[0], 4)) == list(np.around(acc[0], 4))
    assert list(np.around(good_acc[1], 4)) == list(np.around(acc[1], 4))

def test_Track_coords_from_acc_w_yaw(track_data_path):
    t = Track(track_data_path)
    acc, yaw = t.coords_from_acc(ret_yaw=True)

    with open("test_data/acc_coords_w_yaw.pkl", "rb") as FILE:
        data: tuple = pickle.load(FILE)

    good_acc = data[0]
    good_yaw = data[1]
    
    # @TODO 
    # NEED TO FIX
    # np.testing.assert_array_equal(good_acc, acc)
    # np.testing.assert_array_equal(good_yaw, yaw)
    assert list(np.around(good_acc[0], 4)) == list(np.around(acc[0], 4))
    assert list(np.around(good_yaw, 4)) == list(np.around(yaw, 4))

def test_Track_radius_from_gps_coords(track_data_path):
    t = Track(track_data_path)
    gps = t.coords_from_gps()
    
    rads = t.radius_from_gps_coords(gps)

    with open("test_data/rads_data.pkl", "rb") as FILE:
        good_rads = pickle.load(FILE)

    # np.testing.assert_array_equal(good_rads, rads)

    #assert list(np.around(good_rads, 1)) == list(np.around(rads, 1))

def test_Track_get_lap_bounds(track_data_path):
    t = Track(track_data_path)
    bounds = t.get_lap_bounds(1)
    
    assert bounds == (77, 14099)

def test_Track_get_lap_bounds_none(track_data_path):
    t = Track(track_data_path)
    bounds = t.get_lap_bounds(2)
    
    assert bounds == None


# def test_plot_track_full(track_data_path):
#     t = Track(track_data_path)

#     t.plot_track()


# def test_plot_track_bounded(track_data_path):
#     t = Track(track_data_path)

#     t.plot_track(t_bound=(0, 100))


# def test_plot_track_missing_lat(track_data_path):   NEED TO GET FAKE DATA
#     t = Track(track_data_path)

#     with pytest.raises(VariableNotPresent) as e_info:
#         t.plot_track()


# def test_plot_track_missing_long(track_data_path):
#     t = Track(track_data_path)

#     with pytest.raises(VariableNotPresent) as e_info:
#         t.plot_track()

# def test_plot_track_missing_xtime(track_data_path):
#     t = Track(track_data_path)

#     with pytest.raises(VariableNotPresent) as e_info:
#         t.plot_track(t_bound=(0,100))
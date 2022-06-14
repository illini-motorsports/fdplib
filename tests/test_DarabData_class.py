from conftest import darab_data_simple
import numpy as np
import fdplib.darab as darab
import pickle

def test_DarabData_loading(darab_data_simple: darab.DarabData):
    assert darab_data_simple.get_var("xtime") != None


def test_DarabData_get_var(darab_data_simple: darab.DarabData):
    xtime = darab_data_simple.get_var("xtime")

    assert len(xtime) == 94
    assert type(xtime) == list
    assert xtime[0] == 0

def test_DarabData_get_var_as_timeseries(darab_data_simple: darab.DarabData):
    xtime = darab_data_simple.get_var("xtime", timeseries=True)
    
    assert len(xtime) == 2
    assert len(xtime[0]) == 94
    assert len(xtime[1]) == 94


def test_DarabData_get_var_np(darab_data_simple: darab.DarabData):
    xtime = darab_data_simple.get_var_np("xtime")

    assert type(xtime) == np.ndarray
    assert xtime.shape == (94,)


def test_DarabData_get_var_np_as_timeseries(darab_data_simple: darab.DarabData):
    xtime = darab_data_simple.get_var_np("xtime", timeseries=True)

    assert type(xtime) == np.ndarray
    assert xtime.shape == (2,94)


def test_DarabData_bracket_accessor(darab_data_simple: darab.DarabData):
    assert darab_data_simple["xtime"] == darab_data_simple.get_var("xtime")


def test_DarabData_list_list_vars(darab_data_simple: darab.DarabData):
    with open("test_data/simple_test_data_labels.pkl", "rb") as FILE:
        labels: list = pickle.load(FILE)

    assert darab_data_simple.list_vars() == labels

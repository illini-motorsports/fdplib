from conftest import darab_data_simple
import fdplib.darab as darab

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
    xtime = darab_data_simple.get_var("xtime", timeseries=True)
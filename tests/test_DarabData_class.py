import fdplib.darab as darab

def test_DarabData_loading():
    d = darab.DarabData("test_data/simple_test_data.txt")
    assert d.get_var("xtime") != None

def test_DarabData_get_var():
    d = darab.DarabData("test_data/simple_test_data.txt")

    xtime = d.get_var("xtime")

    assert len(xtime) == 94
    assert type(xtime) == list
    assert xtime[0] == 0
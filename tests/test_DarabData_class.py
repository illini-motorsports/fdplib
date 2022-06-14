import fdplib.darab as darab

def test_DarabData_loading():
    d = darab.DarabData("test_data/simple_test_data.txt")
    assert d.get_var("xtime") != None
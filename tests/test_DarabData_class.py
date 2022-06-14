import fdplib.darab as darab

def test_DarabData_loading():
    d = darab.DarabData("test_data/example.txt")
    assert d.get_var("xtime") != None
import pytest
import fdplib.darab as darab

@pytest.fixture
def darab_data_simple() -> darab.DarabData:
    return darab.DarabData("test_data/simple_test_data.txt")
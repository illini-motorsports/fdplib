import fdplib.errors as errors

def test_FileDoesNotExist():
    er = errors.FileDoesNotExist("testfile.txt")

    assert str(er) == f"Filepath \"testfile.txt\" does not exist or could not be opened"

def test_VariableNotPresent():
    er = errors.VariableNotPresent("xtime")

    assert str(er) == "Variable \"xtime\" does not exist in the dataset"
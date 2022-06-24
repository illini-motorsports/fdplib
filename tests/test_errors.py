import fdplib.errors as errors

def test_FileDoesNotExist():
    er = errors.FileDoesNotExist("testfile.txt")

    assert str(er) == f"file \"testfile.txt\" does not exist or could not be opened"
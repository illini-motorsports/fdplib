import fdplib.errors as errors

def test_FileDoesNotExist():
    er = errors.FileDoesNotExist("testfile.txt")

    assert str(er) == f"Filepath \"testfile.txt\" does not exist or could not be opened"
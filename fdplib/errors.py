class FileDoesNotExist(Exception):
    
    def __init__(self, filename: str) -> None:
        self.filename = filename

    def __str__(self):
        return f"Filepath \"{self.filename}\" does not exist or could not be opened"


class VariableNotPresent(Exception):
    
    def __init__(self, var_name: str) -> None:
        self.var_name = var_name

    def __str__(self):
        return f"Variable \"{self.var_name}\" does not exist in the dataset"
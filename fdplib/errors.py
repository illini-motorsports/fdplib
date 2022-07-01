class FileDoesNotExist(Exception):
    
    def __init__(self, filename: str) -> None:
        self.filename = filename

    def __str__(self):
        return f"Filepath \"{self.filename}\" does not exist or could not be opened"
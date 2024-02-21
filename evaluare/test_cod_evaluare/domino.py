class Domino:
    def __init__(self, domino_file: str, pos1: str, pos2: str, val1: int, val2: int) -> None:
        self.domino_file = domino_file
        self.pos1 = pos1
        self.pos2 = pos2
        self.val1 = val1
        self.val2 = val2

    def __str__(self) -> str:
        return f"{self.domino_file} {self.pos1} {self.pos2} {self.val1} {self.val2}"
    
    def __repr__(self) -> str:
        return str(self)
    
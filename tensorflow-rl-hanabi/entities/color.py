class Color:
    def __init__(self, color_id: int, name: str):
        self.color_id = color_id
        self.name = name

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return self.color_id == other.color_id

    def __hash__(self):
        return hash(repr(self))


BLUE = Color(0, 'Blue')
GREEN = Color(1, 'Green')

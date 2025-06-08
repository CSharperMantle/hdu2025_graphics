from const import *
from model import *
from render import *
from type import *


class BlockView:
    COLORS: dict[TetrominoShape, ColorRGBf] = {
        TetrominoShape.I: COLOR_CYAN,
        TetrominoShape.O: COLOR_YELLOW,
        TetrominoShape.T: COLOR_ORANGE,
        TetrominoShape.L: COLOR_BLUE_LIGHTER,
        TetrominoShape.S: COLOR_GREEN,
    }

    _pos: VecXZYi
    _type: TetrominoShape
    _alpha: float

    @property
    def pos(self) -> VecXZYi:
        return self._pos

    @property
    def color(self) -> ColorRGBf:
        return self.COLORS[self._type]

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        self._alpha = min(1.0, max(0.0, value))

    def __init__(self, pos: VecXZYi, type: TetrominoShape, alpha: float):
        self._pos = pos
        self._type = type
        self.alpha = alpha

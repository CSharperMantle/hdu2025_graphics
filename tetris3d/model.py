import typing as ty
from copy import deepcopy
from enum import Enum, IntEnum, auto

import numpy as np
import numpy.typing as npt

Position = tuple[int, int, int]  # (x, z, y)


class TetrominoShape(IntEnum):
    I = auto()
    O = auto()
    T = auto()
    L = auto()
    J = auto()
    S = auto()
    Z = auto()


class Axis(IntEnum):
    X = auto()
    Y = auto()
    Z = auto()


class MoveDir(IntEnum):
    X_POS = auto()
    X_NEG = auto()
    Z_POS = auto()
    Z_NEG = auto()
    Y_NEG = auto()


class Tetromino:
    SHAPES: dict[TetrominoShape, list[Position]] = {
        TetrominoShape.I: [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)],
        TetrominoShape.O: [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],
        TetrominoShape.T: [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)],
        TetrominoShape.L: [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
        TetrominoShape.J: [(0, 0, 0), (-1, 0, 0), (0, 1, 0), (0, 0, 1)],
        TetrominoShape.S: [(0, 0, 0), (-1, 0, 0), (0, 1, 0), (1, 1, 0)],
        TetrominoShape.Z: [(0, 0, 0), (1, 0, 0), (0, 1, 0), (-1, 1, 0)],
    }

    _blocks: npt.NDArray[np.int_]
    _position: npt.NDArray[np.int_]

    def __init__(self, shape: TetrominoShape, position: Position):
        self._blocks = np.array(self.SHAPES[shape], dtype=int)
        self._position = np.array(position, dtype=int)

    def rotate(self, axis: Axis):
        match axis:
            case Axis.X:
                mat = np.array(
                    [
                        [1, 0, 0],
                        [0, 0, -1],
                        [0, 1, 0],
                    ],
                    dtype=int,
                )
            case Axis.Y:
                mat = np.array(
                    [
                        [0, 0, 1],
                        [0, 1, 0],
                        [-1, 0, 0],
                    ],
                    dtype=int,
                )
            case Axis.Z:
                mat = np.array(
                    [
                        [0, -1, 0],
                        [1, 0, 0],
                        [0, 0, 1],
                    ],
                    dtype=int,
                )
            case v:
                raise ValueError(f"Invalid axis: {v}")
        self._blocks = np.dot(self._blocks, mat)

    def move(self, direction: MoveDir):
        match direction:
            case MoveDir.X_POS:
                self._position[0] += 1
            case MoveDir.X_NEG:
                self._position[0] -= 1
            case MoveDir.Z_POS:
                self._position[1] += 1
            case MoveDir.Z_NEG:
                self._position[1] -= 1
            case MoveDir.Y_NEG:
                self._position[2] -= 1

    @property
    def world_blocks(self) -> ty.Iterator[Position]:
        for block in self._blocks:
            yield tuple(self._position + block)


class GameModel:
    _frozen: npt.NDArray[np.bool_]
    _current_piece: ty.Optional[Tetromino]
    _score: int
    _dimensions: tuple[int, int, int]

    def __init__(self, width: int, depth: int, height: int):
        self._frozen = np.zeros((width, depth, height), dtype=bool)
        self._current_piece = None
        self._score = 0
        self._dimensions = (width, depth, height)

    def _check_collision(self, piece: Tetromino) -> bool:
        for block in piece.world_blocks:
            if (
                block[0] < 0
                or block[0] >= self._dimensions[0]
                or block[1] < 0
                or block[1] >= self._dimensions[1]
                or block[2] < 0
                or block[2] >= self._dimensions[2]
            ):
                return True
            if self._frozen[block]:
                return True
        return False

    def _freeze_current_piece(self):
        assert self._current_piece is not None

        for block in self._current_piece.world_blocks:
            self._frozen[block] = True
            pass

    def _clear_planes(self):
        filled_layers = set(
            y
            for y in reversed(range(self._dimensions[2]))
            if self._frozen[:, :, y].all()
        )
        new_blocks = np.zeros_like(self._frozen)
        new_y = 0
        for y in range(self._dimensions[2]):
            if y not in filled_layers:
                new_blocks[:, :, new_y] = self._frozen[:, :, y]
                new_y += 1
        self._frozen = new_blocks
        self._score += len(filled_layers)

    def spawn_piece(self, shape: TetrominoShape):
        assert self._current_piece is None

        piece = Tetromino(
            shape,
            (
                self._dimensions[0] // 2,
                self._dimensions[1] // 2,
                self._dimensions[2] - 4,
            ),
        )
        self._current_piece = piece

    def move(self, direction: MoveDir):
        assert self._current_piece is not None

        piece_clone = deepcopy(self._current_piece)
        piece_clone.move(direction)
        if not self._check_collision(piece_clone):
            self._current_piece = piece_clone
        else:
            if direction == MoveDir.Y_NEG:
                self._freeze_current_piece()

    def rotate(self, axis: Axis):
        assert self._current_piece is not None

        self._current_piece.rotate(axis)

    @property
    def all_blocks(self) -> ty.Iterator[Position]:
        if self._current_piece is not None:
            for block in self._current_piece.world_blocks:
                yield block
        with np.nditer(self._frozen, flags=["multi_index"]) as it:
            for block in it:
                if block:
                    yield (it.multi_index[0], it.multi_index[1], it.multi_index[2])

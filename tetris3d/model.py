import typing as ty
from copy import deepcopy
from enum import IntEnum, auto

import numpy as np
import numpy.typing as npt
from type import *


class TetrominoShape(IntEnum):
    I = auto()
    O = auto()
    T = auto()
    L = auto()
    S = auto()


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


def _get_dims(blocks: npt.NDArray[np.int_]) -> VecXZYi:
    if len(blocks) == 0:
        return (0, 0, 0)
    min_coords = blocks.min(axis=0)
    max_coords = blocks.max(axis=0)
    dims = max_coords - min_coords + 1
    return (int(dims[0]), int(dims[1]), int(dims[2]))


class Tetromino:
    SHAPES: dict[TetrominoShape, list[VecXZYi]] = {
        TetrominoShape.I: [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)],
        TetrominoShape.O: [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],
        TetrominoShape.T: [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0)],
        TetrominoShape.L: [(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 0, 1)],
        TetrominoShape.S: [(0, 0, 0), (-1, 0, 0), (0, 1, 0), (1, 1, 0)],
    }

    DIMS: dict[TetrominoShape, VecXZYi] = {
        shape: _get_dims(np.array(blocks, dtype=int)) for shape, blocks in SHAPES.items()
    }

    WALL_KICK_OFFSETS: list[VecXZYi] = [
        (1, 0, 0),  # Right
        (-1, 0, 0),  # Left
        (0, 1, 0),  # Forward (Z+)
        (0, -1, 0),  # Back (Z-)
        (0, 0, 1),  # Up
        (1, 1, 0),  # Right-Forward
        (-1, 1, 0),  # Left-Forward
        (1, -1, 0),  # Right-Back
        (-1, -1, 0),  # Left-Back
    ]

    ROTATE_MATRIX: dict[Axis, npt.NDArray[np.int_]] = {
        Axis.X: np.array(
            [
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, 0],
            ],
            dtype=int,
        ),
        Axis.Y: np.array(
            [
                [0, 0, 1],
                [0, 1, 0],
                [-1, 0, 0],
            ],
            dtype=int,
        ),
        Axis.Z: np.array(
            [
                [0, -1, 0],
                [1, 0, 0],
                [0, 0, 1],
            ],
            dtype=int,
        ),
    }

    _blocks: npt.NDArray[np.int_]
    _position: npt.NDArray[np.int_]
    _shape: TetrominoShape

    @property
    def world_blocks(self) -> ty.Iterator[tuple[VecXZYi, TetrominoShape]]:
        for block in self._blocks:
            pos = self._position + block
            yield ((int(pos[0]), int(pos[1]), int(pos[2])), self._shape)

    @property
    def shape(self) -> TetrominoShape:
        return self._shape

    def __init__(self, shape: TetrominoShape, position: VecXZYi):
        self._blocks = np.array(self.SHAPES[shape], dtype=int)
        self._position = np.array(position, dtype=int)
        self._shape = shape

    def rotate(self, axis: Axis):
        if len(self._blocks) > 0:
            center = np.mean(self._blocks, axis=0, dtype=float)
            center = np.round(center).astype(int)
        else:
            center = np.array([0, 0, 0], dtype=int)
        mat = Tetromino.ROTATE_MATRIX[axis]
        self._blocks = self._blocks - center
        self._blocks = np.dot(self._blocks, mat)
        self._blocks = self._blocks + center

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


class GameModel:
    _frozen: npt.NDArray[np.bool_]
    _frozen_types: npt.NDArray[np.uint8]
    _current_piece: ty.Optional[Tetromino]
    _score: int
    _dims: VecXZYi
    _on_frozen: ty.Callable[[ty.Self], tuple[TetrominoShape, ty.Iterable[Axis]]]
    _on_score: ty.Callable[[ty.Self], None]
    _on_failure: ty.Callable[[ty.Self], None]

    @property
    def score(self) -> int:
        return self._score

    @property
    def dims(self) -> VecXZYi:
        return self._dims

    @property
    def all_blocks(self) -> ty.Iterator[tuple[VecXZYi, TetrominoShape]]:
        if self._current_piece is not None:
            for block in self._current_piece.world_blocks:
                yield block
        with np.nditer(self._frozen, flags=["multi_index"]) as it:
            for block in it:
                if block:
                    yield (
                        (int(it.multi_index[0]), int(it.multi_index[1]), int(it.multi_index[2])),
                        self._frozen_types[it.multi_index],
                    )

    @property
    def active_blocks(self) -> ty.Iterator[tuple[VecXZYi, TetrominoShape]]:
        if self._current_piece is not None:
            for block in self._current_piece.world_blocks:
                yield block
        else:
            return

    def __init__(
        self,
        width: int,
        depth: int,
        height: int,
        on_frozen: ty.Callable[[ty.Self], tuple[TetrominoShape, ty.Iterable[Axis]]] = lambda _: (
            TetrominoShape.I,
            (),
        ),
        on_score: ty.Callable[[ty.Self], None] = lambda _: None,
        on_failure: ty.Callable[[ty.Self], None] = lambda _: None,
    ):
        self._frozen = np.zeros((width, depth, height), dtype=bool)
        self._frozen_types = np.zeros((width, depth, height), dtype=np.uint8)
        self._current_piece = None
        self._score = 0
        self._on_frozen = on_frozen
        self._on_score = on_score
        self._on_failure = on_failure
        self._dims = (width, depth, height)

    def _check_collision(self, piece: Tetromino) -> bool:
        for block, _ in piece.world_blocks:
            if (
                block[0] < 0
                or block[0] >= self._dims[0]
                or block[1] < 0
                or block[1] >= self._dims[1]
                or block[2] < 0
                or block[2] >= self._dims[2]
            ):
                return True
            if self._frozen[block]:
                return True
        return False

    def _clear_planes(self):
        filled_layers = set(
            y for y in reversed(range(self._dims[2])) if self._frozen[:, :, y].all()
        )
        score = len(filled_layers)
        if score == 0:
            return
        new_blocks = np.zeros_like(self._frozen)
        new_types = np.zeros_like(self._frozen_types)
        new_y = 0
        for y in range(self._dims[2]):
            if y not in filled_layers:
                new_blocks[:, :, new_y] = self._frozen[:, :, y]
                new_types[:, :, new_y] = self._frozen_types[:, :, y]
                new_y += 1
        self._frozen = new_blocks
        self._frozen_types = new_types
        self._score += score
        self._on_score(self)

    def _freeze_current_piece(self):
        if self._current_piece is None:
            return
        for block, type in self._current_piece.world_blocks:
            self._frozen[block] = True
            self._frozen_types[block] = type
        self._current_piece = None
        piece, rotation_plan = self._on_frozen(self)
        self.spawn_piece(piece)
        for axis in rotation_plan:
            if not self.rotate(axis):
                break

    def spawn_piece(self, shape: TetrominoShape):
        if self._current_piece is not None:
            return
        dims = Tetromino.DIMS[shape]
        piece = Tetromino(
            shape,
            (
                (self._dims[0] - dims[0]) // 2,
                (self._dims[1] - dims[1]) // 2,
                self._dims[2] - dims[2] - 1,
            ),
        )
        if self._check_collision(piece):
            self._on_failure(self)
            return
        self._current_piece = piece

    def move(self, direction: MoveDir) -> bool:
        if self._current_piece is None:
            return False
        piece_clone = deepcopy(self._current_piece)
        piece_clone.move(direction)
        if not self._check_collision(piece_clone):
            self._current_piece = piece_clone
            return True
        if direction == MoveDir.Y_NEG:
            self._freeze_current_piece()
        return False

    def rotate(self, axis: Axis) -> bool:
        if self._current_piece is None:
            return False
        original_piece = deepcopy(self._current_piece)
        self._current_piece.rotate(axis)
        if not self._check_collision(self._current_piece):
            return True
        for kick in Tetromino.WALL_KICK_OFFSETS:
            kick_test = deepcopy(self._current_piece)
            kick_test._position += np.array(kick, dtype=int)
            if not self._check_collision(kick_test):
                self._current_piece = kick_test
                return True
        self._current_piece = original_piece
        return False

    def update(self):
        self._clear_planes()
        if self._current_piece is None:
            return
        self.move(MoveDir.Y_NEG)

    def drop(self):
        if self._current_piece is None:
            return
        while self.move(MoveDir.Y_NEG):
            pass

    def get_height_below_for(self, pos: VecXZYi) -> int:
        if not (0 <= pos[0] < self._dims[0] and 0 <= pos[1] < self._dims[1]):
            return 0
        height = 0
        for y in range(pos[2], -1, -1):
            if self._frozen[pos[0], pos[1], y]:
                break
            height += 1
        return height

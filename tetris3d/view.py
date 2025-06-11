# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Rong Bao <webmaster@csmantle.top>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

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

GAME_AREA_SIZE = (6, 6, 18)
INITIAL_WINDOW_SIZE = (800, 600)
YAW_SENSITIVITY = 0.3
PITCH_SENSITIVITY = 0.3
SCROLL_SENSITIVITY = 1.0
PAN_SENSITIVITY = 0.5
FRAME_PERIOD_MS = 1
GAME_TICK_PERIOD_MS = 2000
SELECTION_ANIMATION_DURATION_US = 200_000

INITIAL_DISTANCE = 30.0
INITIAL_YAW = 45.0
INITIAL_PITCH = 30.0

VERTEX_ORIGIN = (0.0, 0.0, 0.0)
RENDER_BLOCK_SIZE = 1.0
RENDER_BLOCK_GAP = 0.002
RENDER_UNSELECTED_ALPHA = 0.1

COLOR_CYAN = (0.0, 1.0, 1.0)
COLOR_YELLOW = (1.0, 1.0, 0.0)
COLOR_ORANGE = (1.0, 0.5, 0.0)
COLOR_PURPLE = (0.5, 0.0, 0.5)
COLOR_RED = (1.0, 0.0, 0.0)
COLOR_GREEN = (0.0, 1.0, 0.0)
COLOR_BLUE = (0.0, 0.0, 1.0)
COLOR_BLUE_LIGHTER = (0.0, 0.5, 1.0)
COLOR_WHITE = (1.0, 1.0, 1.0)
COLOR_GRAY_LIGHTER = (0.5, 0.5, 0.5)

LIGHT0_POSITION = (15.0, 20.0, 10.0, 1.0)
LIGHT0_AMBIENT = (0.15, 0.14, 0.13, 1.0)
LIGHT0_DIFFUSE = (0.95, 0.92, 0.85, 1.0)
LIGHT0_SPECULAR = (0.7, 0.7, 0.7, 1.0)
LIGHT1_POSITION = (-8.0, 5.0, -12.0, 1.0)
LIGHT1_AMBIENT = (0.0, 0.0, 0.0, 1.0)
LIGHT1_DIFFUSE = (0.3, 0.3, 0.4, 1.0)
LIGHT1_SPECULAR = (0.1, 0.1, 0.1, 1.0)
LIGHT_MODEL_LOCAL_VIEWER = (1.0,)

MATERIAL_AMBIENT = (0.15, 0.15, 0.15, 1.0)
MATERIAL_DIFFUSE = (0.7, 0.7, 0.7, 1.0)
MATERIAL_SPECULAR = (0.5, 0.5, 0.5, 1.0)
MATERIAL_SHININESS = (48.0,)

MATERIAL_NULL = (0.0, 0.0, 0.0, 1.0)

PATH_TEXTURE_BLOCK = "./tetris3d/assets/texture_block.bmp"

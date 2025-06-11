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

import math
import typing as ty

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

Point = tuple[float, float]


def basis(t: float, n: int, i: int) -> float:
    return (
        (math.factorial(n) / (math.factorial(i) * math.factorial(n - i)))
        * (t**i)
        * ((1 - t) ** (n - i))
    )


def bezier(t: float, points: ty.Sequence[Point]) -> Point:
    n = len(points) - 1
    x = sum(basis(t, n, i) * points[i][0] for i in range(n + 1))
    y = sum(basis(t, n, i) * points[i][1] for i in range(n + 1))
    return x, y


STEPS = 10000
POINTS = [(0, 0), (300, 0), (0, 200), (300, 300)]

points = [bezier(i / STEPS, POINTS) for i in range(STEPS)]


def display():
    glClear(GL_COLOR_BUFFER_BIT)
    glColor3f(1.0, 0.0, 0.0)
    glBegin(GL_POINTS)
    for [x, y] in points:
        glVertex2f(x, y)
    glEnd()
    glFlush()


def reshape(w: float, h: float):
    glViewport(0, 0, int(w), int(h))
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0.0, w, 0.0, h)


glutInit(sys.argv)
glutInitWindowSize(400, 300)
glutInitWindowPosition(100, 100)
glutCreateWindow(b"OpenGL Test")
glutDisplayFunc(display)
glutReshapeFunc(reshape)
glutMainLoopEvent()
input("Press Enter to exit")

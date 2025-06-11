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

import array
import functools as ft
import logging
import math
import typing as ty

import OpenGL.GL as gl
import OpenGL.GLU as glu
import OpenGL.GLUT as glut

Point: ty.TypeAlias = array.ArrayType

INTRO = """Interactive Bezier curve demo

CONTROLS
* Left mouse button: Add point to the current segment/spline.
* Middle mouse button: Finish the current segment, then create a new segment starting from the last point.
* Right mouse button: Remove the last point from current segment.
* G: Squeeze segments, removing all empty or 1-membered segments.
* ESC: Quit."""

BEZIER_STEPS = 1000
POINT_COLOR = (255, 0, 0)
BEZIER_COLOR = (0, 0, 255)
CONTROL_COLOR_EVEN = (144, 238, 144)
CONTROL_COLOR_ODD = (1, 50, 32)
SELECT_RADIUS_PX = 10

segments: list[list[Point]] = []
modify_point: ty.Optional[Point] = None


@ft.cache
def basis(t: float, n: int, i: int) -> float:
    return (
        (math.factorial(n) / (math.factorial(i) * math.factorial(n - i)))
        * (t**i)
        * ((1 - t) ** (n - i))
    )


def bezier(t: float, points: ty.Sequence[Point]) -> Point:
    n = len(points) - 1
    x = sum((basis(t, n, i) * points[i][0] for i in range(n + 1)))
    y = sum((basis(t, n, i) * points[i][1] for i in range(n + 1)))
    return array.array("d", (x, y))


def draw_point(point: Point, size: float = 5.0):
    x, y = point
    gl.glColor3ub(*POINT_COLOR)
    gl.glBegin(gl.GL_POLYGON)
    for i in range(32):
        angle = 2.0 * math.pi * i / 32
        gl.glVertex2f(x + size * math.cos(angle), y + size * math.sin(angle))
    gl.glEnd()


def draw_control_points(points: ty.Sequence[Point]):
    for p in points:
        draw_point(p)


def draw_control_polygon(points: ty.Sequence[Point], color: tuple[float, float, float]):
    if len(points) < 2:
        return
    gl.glColor3ub(*color)
    gl.glBegin(gl.GL_LINE_STRIP)
    for p in points:
        gl.glVertex2f(*p)
    gl.glEnd()


def draw_bezier_curve(points: ty.Sequence[Point]):
    if len(points) < 2:
        return
    gl.glColor3ub(*BEZIER_COLOR)
    gl.glBegin(gl.GL_LINE_STRIP)
    for i in range(BEZIER_STEPS + 1):
        t = i / BEZIER_STEPS
        gl.glVertex2f(*bezier(t, points))
    gl.glEnd()


def handle_display():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    for i, segment in enumerate(segments):
        draw_control_points(segment)
        draw_control_polygon(
            segment, CONTROL_COLOR_EVEN if i % 2 == 0 else CONTROL_COLOR_ODD
        )
        draw_bezier_curve(segment)
    glut.glutSwapBuffers()


def handle_reshape(width: int, height: int):
    gl.glViewport(0, 0, width, height)
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    glu.gluOrtho2D(0, width, height, 0)
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()


def handle_mouse(button: int, state: int, x: int, y: int):
    global modify_point
    if button == glut.GLUT_LEFT_BUTTON and state == glut.GLUT_DOWN:
        for segment in segments:
            for point in segment:
                if (point[0] - x) ** 2 + (point[1] - y) ** 2 < SELECT_RADIUS_PX**2:
                    modify_point = point
                    logging.info(f"Selected point {tuple(point)}")
                    return
        if len(segments) == 0:
            segments.append([])
        segment_id = len(segments) - 1
        p = array.array("d", (float(x), float(y)))
        logging.info(f"({p[0]}, {p[1]}) -> segment {segment_id}")
        segments[segment_id].append(p)
        glut.glutPostRedisplay()
    elif button == glut.GLUT_RIGHT_BUTTON and state == glut.GLUT_DOWN:
        if len(segments) == 0:
            return
        segment_id = len(segments) - 1
        segment = segments[segment_id]
        if len(segment) >= 1:
            p = segment.pop()
            logging.info(f"({p[0]}, {p[1]}) <- segment {segment_id}")
            if len(segment) <= 1:
                segments.pop()
                logging.info(f"Delete segment {segment_id}")
            glut.glutPostRedisplay()
    elif button == glut.GLUT_MIDDLE_BUTTON and state == glut.GLUT_DOWN:
        if len(segments) > 0 and len(segments[-1]) > 0:
            segments.append([segments[-1][-1]])
            logging.info(f"Create segment {len(segments) - 1} connected with previous")
        else:
            segments.append([])
            logging.info(f"Create segment {len(segments) - 1}")
    elif button == glut.GLUT_LEFT_BUTTON and state == glut.GLUT_UP:
        if modify_point is not None:
            modify_point = None
            logging.debug("Selection released")


def handle_motion(x: int, y: int):
    if modify_point is not None:
        modify_point[0] = float(x)
        modify_point[1] = float(y)
        glut.glutPostRedisplay()


def handle_keyboard(key: bytes, x, y):
    if key == b"\x1b":
        glut.glutDestroyWindow(glut.glutGetWindow())
    elif key == b"g":
        global segments
        empty_seg_id = [i for i, seg in enumerate(segments) if len(seg) <= 1]
        new_segs = []
        for i in range(len(segments)):
            if i in empty_seg_id:
                logging.info(f"Squeezed out segment {i} with {len(segments[i])} points")
            else:
                new_segs.append(segments[i])
        segments = new_segs
        glut.glutPostRedisplay()
    elif key == b"c":
        logging.info(f"Basis function cache: {basis.cache_info()}")


def main():
    print(INTRO)
    logging.basicConfig(level=logging.DEBUG)
    glut.glutInit()
    glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA)
    glut.glutInitWindowSize(800, 600)
    glut.glutCreateWindow(b"Interactive Bezier curve demo")
    glut.glutDisplayFunc(handle_display)
    glut.glutReshapeFunc(handle_reshape)
    glut.glutMouseFunc(handle_mouse)
    glut.glutMotionFunc(handle_motion)
    glut.glutKeyboardFunc(handle_keyboard)
    gl.glClearColor(1.0, 1.0, 1.0, 1.0)
    gl.glEnable(gl.GL_LINE_SMOOTH)
    gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    gl.glLineWidth(2.0)
    glut.glutMainLoop()


if __name__ == "__main__":
    main()

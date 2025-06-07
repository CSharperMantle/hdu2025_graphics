import logging
import math
import random
import sys
import typing as ty
import itertools as it
import time

from PIL import Image
import OpenGL.GL as gl
import OpenGL.GLU as glu
import OpenGL.GLUT as glut

from model import Axis, GameModel, MoveDir, TetrominoShape
from const import *
from render import *

window_size = INITIAL_WINDOW_SIZE
game = GameModel(*GAME_AREA_SIZE)

camera_distance = 30.0
camera_yaw = 45.0
camera_pitch = 30.0
mouse_last_pos = (0, 0)
mouse_lb_down = False

game_center = (game.dims[0] / 2.0, game.dims[2] / 2.0, game.dims[1] / 2.0)

block_renderer: ty.Optional[BlockRenderer] = None
marker_renderer: ty.Optional[MarkerRenderer] = None


def draw_axes():
    gl.glLineWidth(1.0)

    gl.glColor3f(*COLOR_RED)
    gl.glBegin(gl.GL_LINE_STRIP)
    gl.glVertex3f(*VERTEX_ORIGIN)
    gl.glVertex3f(game.dims[0], 0.0, 0.0)
    gl.glVertex3f(game.dims[0], 0.0, game.dims[1])
    gl.glVertex3f(0.0, 0.0, game.dims[1])
    gl.glVertex3f(*VERTEX_ORIGIN)
    gl.glEnd()

    gl.glColor3f(*COLOR_GREEN)
    gl.glBegin(gl.GL_LINES)
    gl.glVertex3f(*VERTEX_ORIGIN)
    gl.glVertex3f(0.0, game.dims[2], 0.0)
    gl.glVertex3f(game.dims[0], 0.0, 0.0)
    gl.glVertex3f(game.dims[0], game.dims[2], 0.0)
    gl.glVertex3f(game.dims[0], 0.0, game.dims[1])
    gl.glVertex3f(game.dims[0], game.dims[2], game.dims[1])
    gl.glVertex3f(0.0, 0.0, game.dims[1])
    gl.glVertex3f(0.0, game.dims[2], game.dims[1])
    gl.glEnd()

    gl.glColor3f(*COLOR_BLUE)
    gl.glBegin(gl.GL_LINE_STRIP)
    gl.glVertex3f(0.0, game.dims[2], 0.0)
    gl.glVertex3f(game.dims[0], game.dims[2], 0.0)
    gl.glVertex3f(game.dims[0], game.dims[2], game.dims[1])
    gl.glVertex3f(0.0, game.dims[2], game.dims[1])
    gl.glVertex3f(0.0, game.dims[2], 0.0)
    gl.glEnd()

    assert marker_renderer is not None
    for x, z, y in it.product(
        range(GAME_AREA_SIZE[0]), range(GAME_AREA_SIZE[1]), range(GAME_AREA_SIZE[2])
    ):
        marker_renderer.render((float(x), float(y), float(z)))

    gl.glLineWidth(1.0)


def draw_blocks():
    for block in game.all_blocks:
        assert block_renderer is not None
        block_renderer.render((float(block[0]), float(block[2]), float(block[1])), COLOR_WHITE)


def repose_camera():
    gl.glLoadIdentity()
    eye_x = game_center[0] + camera_distance * math.cos(math.radians(camera_pitch)) * math.sin(
        math.radians(camera_yaw)
    )
    eye_y = game_center[1] + camera_distance * math.sin(math.radians(camera_pitch))
    eye_z = game_center[2] + camera_distance * math.cos(math.radians(camera_pitch)) * math.cos(
        math.radians(camera_yaw)
    )
    glu.gluLookAt(
        eye_x,
        eye_y,
        eye_z,
        *game_center,
        0.0,
        1.0,
        0.0,
    )


def handle_display():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glMatrixMode(gl.GL_MODELVIEW)
    repose_camera()
    draw_axes()
    draw_blocks()
    glut.glutSwapBuffers()


def handle_reshape(width: int, height: int):
    global window_size

    window_size = (width, height)
    gl.glViewport(0, 0, width, height)
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    aspect = width / max(1, height)
    glu.gluPerspective(45.0, aspect, 1.0, 100.0)
    gl.glMatrixMode(gl.GL_MODELVIEW)


def handle_keyboard(key: bytes, x: int, y: int):
    need_redraw = False

    if key == b"\x1b":
        glut.glutDestroyWindow(glut.glutGetWindow())
        return
    elif key == b"s":
        game.move(MoveDir.Y_NEG)
        need_redraw = True
    elif key == b" ":
        game.rotate(Axis.Y)
        need_redraw = True

    if need_redraw:
        glut.glutPostRedisplay()


def handle_mouse(button: int, state: int, x: int, y: int):
    global mouse_last_pos, mouse_lb_down

    if button == glut.GLUT_LEFT_BUTTON:
        if state == glut.GLUT_DOWN:
            mouse_lb_down = True
            mouse_last_pos = (x, y)
        elif state == glut.GLUT_UP:
            mouse_lb_down = False
    glut.glutPostRedisplay()


def handle_motion(x: int, y: int):
    global camera_yaw, camera_pitch, mouse_last_pos

    dx = x - mouse_last_pos[0]
    dy = y - mouse_last_pos[1]

    if mouse_lb_down:
        camera_yaw += dx * YAW_SENSITIVITY
        camera_pitch = max(-89.0, min(89.0, camera_pitch + dy * PITCH_SENSITIVITY))
        mouse_last_pos = (x, y)
        glut.glutPostRedisplay()


def handle_wheel(button: int, dir: int, x: int, y: int):
    global camera_distance

    if dir < 0:
        camera_distance = max(1.0, camera_distance - SCROLL_SENSITIVITY)
    elif dir > 0:
        camera_distance += SCROLL_SENSITIVITY
    glut.glutPostRedisplay()


def handle_idle():
    # glut.glutPostRedisplay()
    pass


def main():
    global block_renderer, marker_renderer

    logging.basicConfig(level=logging.DEBUG)
    random.seed(0x0D000721)

    glut.glutInit(sys.argv)
    glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA | glut.GLUT_DEPTH)
    glut.glutInitWindowSize(window_size[0], window_size[1])
    glut.glutCreateWindow(b"Tetris 3D")

    gl.glClearColor(0.1, 0.1, 0.1, 0.0)
    gl.glShadeModel(gl.GL_SMOOTH)
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_LINE_SMOOTH)
    gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    gl.glLineWidth(1.0)

    with Image.open(PATH_TEXTURE_BLOCK) as tex_block_img:
        tex_block_img_width, tex_block_img_height = tex_block_img.size
        tex_block_data = tex_block_img.tobytes("raw", "RGB", 0, -1)
    tex_block = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex_block)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST_MIPMAP_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    gl.glTexImage2D(
        gl.GL_TEXTURE_2D,
        0,
        gl.GL_RGB,
        tex_block_img_width,
        tex_block_img_height,
        0,
        gl.GL_RGB,
        gl.GL_UNSIGNED_BYTE,
        tex_block_data,
    )
    gl.glGenerateMipmap(gl.GL_TEXTURE_2D)

    block_renderer = BlockRenderer(tex_block)
    marker_renderer = MarkerRenderer()

    glut.glutDisplayFunc(handle_display)
    glut.glutReshapeFunc(handle_reshape)
    glut.glutKeyboardFunc(handle_keyboard)
    glut.glutMouseFunc(handle_mouse)
    glut.glutMotionFunc(handle_motion)
    glut.glutMouseWheelFunc(handle_wheel)
    glut.glutIdleFunc(handle_idle)

    handle_reshape(window_size[0], window_size[1])

    game.spawn_piece(TetrominoShape.I)

    glut.glutMainLoop()


if __name__ == "__main__":
    main()

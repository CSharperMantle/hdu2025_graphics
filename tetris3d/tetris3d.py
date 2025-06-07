import itertools as it
import logging
import math
import random
import sys
import time
import typing as ty

import numpy as np
import OpenGL.GL as gl
import OpenGL.GLU as glu
import OpenGL.GLUT as glut
from const import *
from model import Axis, GameModel, MoveDir, TetrominoShape
from PIL import Image
from render import *

VecXYZ = tuple[float, float, float]  # (x, y, z)
Ray = tuple[VecXYZ, VecXYZ]  # (origin, dir)

window_size = INITIAL_WINDOW_SIZE
game = GameModel(
    *GAME_AREA_SIZE,
    lambda _: random.choice(list(TetrominoShape)),
)

camera_distance = INITIAL_DISTANCE
camera_yaw = INITIAL_YAW
camera_pitch = INITIAL_PITCH
camera_pan = (0.0, 0.0, 0.0)
mouse_last_pos = (0, 0)
mouse_lb_down = False

game_center = (game.dims[0] / 2.0, game.dims[2] / 2.0, game.dims[1] / 2.0)

block_renderer: ty.Optional[BlockRenderer] = None
marker_renderer: ty.Optional[MarkerRenderer] = None

last_tick = 0
paused = False


def get_move_dir(yaw: float, key: ty.Literal[b"w", b"a", b"s", b"d"]) -> MoveDir:
    yaw_ = round(yaw)
    if yaw_ >= 315 or yaw_ < 45:
        if key == b"w":
            return MoveDir.Z_NEG
        elif key == b"a":
            return MoveDir.X_NEG
        elif key == b"s":
            return MoveDir.Z_POS
        elif key == b"d":
            return MoveDir.X_POS
    elif 45 <= yaw_ < 135:
        if key == b"w":
            return MoveDir.X_NEG
        elif key == b"a":
            return MoveDir.Z_POS
        elif key == b"s":
            return MoveDir.X_POS
        elif key == b"d":
            return MoveDir.Z_NEG
    elif 135 <= yaw_ < 225:
        if key == b"w":
            return MoveDir.Z_POS
        elif key == b"a":
            return MoveDir.X_POS
        elif key == b"s":
            return MoveDir.Z_NEG
        elif key == b"d":
            return MoveDir.X_NEG
    else:
        if key == b"w":
            return MoveDir.X_POS
        elif key == b"a":
            return MoveDir.Z_NEG
        elif key == b"s":
            return MoveDir.X_NEG
        elif key == b"d":
            return MoveDir.Z_POS


def ray_intersects_cube(
    ray: Ray,
    origin: VecXYZ,
    size: VecXYZ,
) -> bool:
    start = np.array(ray[0], dtype=np.float32)
    dir = np.array(ray[1], dtype=np.float32)
    origin_ = np.array(origin, dtype=np.float32)
    size_ = np.array(size, dtype=np.float32)
    max_bound = origin_ + size_
    mask_zero = np.abs(dir) < 1e-6
    if np.any(mask_zero & ((start < origin_) | (start > max_bound))):
        return False
    dir_safe = np.where(mask_zero, 1.0, dir)
    t1 = np.where(mask_zero, -np.inf, (origin_ - start) / dir_safe)
    t2 = np.where(mask_zero, np.inf, (max_bound - start) / dir_safe)
    t_near = np.minimum(t1, t2)
    t_far = np.maximum(t1, t2)
    t_min = np.max(t_near)
    t_max = np.min(t_far)
    return (t_min <= t_max) and (t_max >= 0)


def get_ray_from_screen(x: int, y: int) -> Ray:
    viewport = gl.glGetIntegerv(gl.GL_VIEWPORT)
    y = viewport[3] - y
    near_point = glu.gluUnProject(x, y, 0.0)
    far_point = glu.gluUnProject(x, y, 1.0)
    ray_origin = np.array(near_point, dtype=np.float32)
    ray_direction = np.array(far_point, dtype=np.float32) - ray_origin
    ray_direction = ray_direction / np.linalg.norm(ray_direction)
    return tuple(ray_origin), tuple(ray_direction)


def find_first_intersection(
    ray: Ray,
) -> ty.Optional[VecXYZ]:
    closest_dist = float("inf")
    closest_block = None
    for block, _ in game.all_blocks:
        pos = (float(block[0]), float(block[2]), float(block[1]))
        size = (RENDER_BLOCK_SIZE, RENDER_BLOCK_SIZE, RENDER_BLOCK_SIZE)
        if ray_intersects_cube(ray, pos, size):
            block_center = (pos[0] + 0.5, pos[1] + 0.5, pos[2] + 0.5)
            dist = np.linalg.norm(np.array(block_center) - np.array(ray[0]))
            if dist < closest_dist:
                closest_dist = dist
                closest_block = block
    return closest_block


def draw_axes():
    gl.glLineWidth(1.0)

    gl.glDisable(gl.GL_LIGHTING)

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

    gl.glEnable(gl.GL_LIGHTING)

    assert marker_renderer is not None
    for x, z, y in it.product(
        range(GAME_AREA_SIZE[0]), range(GAME_AREA_SIZE[1]), range(GAME_AREA_SIZE[2])
    ):
        marker_renderer.render((float(x), float(y), float(z)))

    gl.glLineWidth(1.0)


def draw_blocks():
    for block, type in game.all_blocks:
        assert block_renderer is not None
        block_renderer.render((float(block[0]), float(block[2]), float(block[1])), type)


def repose_camera():
    gl.glLoadIdentity()
    base_x = game_center[0] + camera_distance * math.cos(math.radians(camera_pitch)) * math.sin(
        math.radians(camera_yaw)
    )
    base_y = game_center[1] + camera_distance * math.sin(math.radians(camera_pitch))
    base_z = game_center[2] + camera_distance * math.cos(math.radians(camera_pitch)) * math.cos(
        math.radians(camera_yaw)
    )
    eye_x = base_x + camera_pan[0]
    eye_y = base_y + camera_pan[1]
    eye_z = base_z + camera_pan[2]

    target_x = game_center[0] + camera_pan[0]
    target_y = game_center[1] + camera_pan[1]
    target_z = game_center[2] + camera_pan[2]

    glu.gluLookAt(eye_x, eye_y, eye_z, target_x, target_y, target_z, 0.0, 1.0, 0.0)


def quit() -> ty.NoReturn:  # type: ignore
    logging.info("Goodbye!")
    glut.glutDestroyWindow(glut.glutGetWindow())


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
    global paused

    if paused:
        if key == b"`":
            paused = False
            logging.info("Game unpaused.")
        elif key == b"\x1b":
            quit()
        return

    need_redraw = False

    if key == b"\x1b":
        quit()
    elif key == b"q":
        game.rotate(Axis.Y)
        need_redraw = True
    elif key == b"e":
        game.rotate(Axis.Z)
        need_redraw = True
    elif key in (b"w", b"a", b"s", b"d"):
        game.move(get_move_dir(camera_yaw, key))
        need_redraw = True
    elif key == b" ":
        game.drop()
        need_redraw = True
    elif key == b"`":
        paused = True
        logging.info("Game paused. Press ` to resume.")
        need_redraw = True

    if need_redraw:
        glut.glutPostRedisplay()


def handle_special(key: int, x: int, y: int):
    global camera_pan, paused

    if paused:
        return

    need_redraw = False

    forward_x = math.sin(math.radians(camera_yaw))
    forward_z = math.cos(math.radians(camera_yaw))
    right_x = math.sin(math.radians(camera_yaw + 90))
    right_z = math.cos(math.radians(camera_yaw + 90))

    if key == glut.GLUT_KEY_DOWN:
        camera_pan = (
            camera_pan[0] + forward_x * PAN_SENSITIVITY,
            camera_pan[1],
            camera_pan[2] + forward_z * PAN_SENSITIVITY,
        )
        need_redraw = True
    elif key == glut.GLUT_KEY_UP:
        camera_pan = (
            camera_pan[0] - forward_x * PAN_SENSITIVITY,
            camera_pan[1],
            camera_pan[2] - forward_z * PAN_SENSITIVITY,
        )
        need_redraw = True
    elif key == glut.GLUT_KEY_LEFT:
        camera_pan = (
            camera_pan[0] - right_x * PAN_SENSITIVITY,
            camera_pan[1],
            camera_pan[2] - right_z * PAN_SENSITIVITY,
        )
        need_redraw = True
    elif key == glut.GLUT_KEY_RIGHT:
        camera_pan = (
            camera_pan[0] + right_x * PAN_SENSITIVITY,
            camera_pan[1],
            camera_pan[2] + right_z * PAN_SENSITIVITY,
        )
        need_redraw = True
    elif key == glut.GLUT_KEY_HOME:
        camera_pan = (0.0, 0.0, 0.0)
        need_redraw = True

    if need_redraw:
        glut.glutPostRedisplay()


def handle_mouse(button: int, state: int, x: int, y: int):
    global mouse_last_pos, mouse_lb_down

    if button == glut.GLUT_LEFT_BUTTON:
        if state == glut.GLUT_DOWN:
            mouse_lb_down = True
            mouse_last_pos = (x, y)
            ray_origin, ray_dir = get_ray_from_screen(x, y)
            block = find_first_intersection((ray_origin, ray_dir))
            if block is not None:
                logging.debug(f"Block selected: {block}")
        elif state == glut.GLUT_UP:
            mouse_lb_down = False
    glut.glutPostRedisplay()


def handle_motion(x: int, y: int):
    global camera_yaw, camera_pitch, mouse_last_pos

    dx = x - mouse_last_pos[0]
    dy = y - mouse_last_pos[1]

    if mouse_lb_down:
        camera_yaw = (camera_yaw + dx * YAW_SENSITIVITY) % 360.0
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
    global last_tick

    if paused:
        return
    if time.perf_counter_ns() - last_tick > 1_000_000_000:
        game.update()
        last_tick = time.perf_counter_ns()
        glut.glutPostRedisplay()


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
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
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
    gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_LOD_BIAS, -0.4)

    block_renderer = BlockRenderer(tex_block)
    marker_renderer = MarkerRenderer()

    gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, LIGHT0_POSITION)
    gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, LIGHT0_DIFFUSE)
    gl.glLightfv(gl.GL_LIGHT0, gl.GL_SPECULAR, LIGHT0_SPECULAR)
    gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, LIGHT0_AMBIENT)
    gl.glLightModelfv(gl.GL_LIGHT_MODEL_LOCAL_VIEWER, LIGHT_MODEL_LOCAL_VIEWER)

    gl.glEnable(gl.GL_LIGHTING)
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_COLOR_MATERIAL)
    gl.glEnable(gl.GL_LIGHT0)

    glut.glutDisplayFunc(handle_display)
    glut.glutReshapeFunc(handle_reshape)
    glut.glutKeyboardFunc(handle_keyboard)
    glut.glutSpecialFunc(handle_special)
    glut.glutMouseFunc(handle_mouse)
    glut.glutMotionFunc(handle_motion)
    glut.glutMouseWheelFunc(handle_wheel)
    glut.glutIdleFunc(handle_idle)

    handle_reshape(window_size[0], window_size[1])

    game.spawn_piece(random.choice(list(TetrominoShape)))

    glut.glutMainLoop()


if __name__ == "__main__":
    main()

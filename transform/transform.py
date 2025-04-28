import logging
import math
import random
import sys
import typing as ty

import OpenGL.GL as gl
import OpenGL.GLU as glu
import OpenGL.GLUT as glut

from OBJFileLoader import OBJ

CENTER_POSITION = (0.0, 0.0, 0.0)
UP_VECTOR = (0.0, 1.0, 0.0)

LIGHT_POSITION = (5.0, 5.0, 5.0, 1.0)
LIGHT_AMBIENT = (0.0, 0.0, 0.0, 1.0)
LIGHT_DIFFUSE = (1.0, 1.0, 1.0, 1.0)
LIGHT_SPECULAR = (1.0, 1.0, 1.0, 1.0)

LIGHT_MODEL_LOCAL_VIEWER = (1.0,)

MATERIAL_AMBIENT = (0.1, 0.1, 0.1, 1.0)
MATERIAL_DIFFUSE = (0.4, 0.4, 0.4, 1.0)
MATERIAL_SPECULAR = (0.9, 0.9, 0.9, 1.0)
MATERIAL_SHININESS = (75.0,)

PAN_SENSITIVITY = 0.01
ROTATION_SENSITIVITY = 0.5
SCALE_STEP = 0.01
ZOOM_STEP = 0.2


class ObjectState(ty.NamedTuple):
    object_type: ty.Literal["cube", "sphere", "obj"]
    backing_file_path: ty.Optional[str]
    angle: tuple[float, float, float]
    scale_factor: float
    translation: tuple[float, float, float]
    symmetry: tuple[float, float, float]
    color: tuple[float, float, float]
    show_axes: bool

    @staticmethod
    def initial() -> "ObjectState":
        return ObjectState(
            object_type="cube",
            backing_file_path=None,
            angle=(0.0, 0.0, 0.0),
            scale_factor=1.0,
            translation=(0.0, 0.0, 0.0),
            symmetry=(1.0, 1.0, 1.0),
            color=(0.7, 0.7, 0.9),
            show_axes=True,
        )


edit_mode: ty.Literal["object", "camera"] = "object"
projection: ty.Literal["perspective", "parallel"] = "perspective"
old_objects: list[ObjectState] = []
active_object = ObjectState.initial()
eye_position = (3.0, 3.0, 3.0)

window_size = (800, 600)
mouse_pos = (0, 0)
mouse_lb_down = False
mouse_rb_down = False


def cartesian_to_spherical(x: float, y: float, z: float) -> tuple[float, float, float]:
    r = math.sqrt(x**2 + y**2 + z**2)
    theta = math.atan2(y, x)
    phi = math.acos(z / r)
    return r, theta, phi


def spherical_to_cartesian(
    r: float, theta: float, phi: float
) -> tuple[float, float, float]:
    x = r * math.sin(phi) * math.cos(theta)
    y = r * math.sin(phi) * math.sin(theta)
    z = r * math.cos(phi)
    return x, y, z


def set_projection():
    aspect = window_size[0] / window_size[1]
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    if projection == "perspective":
        glu.gluPerspective(45, aspect, 0.1, 50.0)
    elif projection == "parallel":
        if aspect > 1.0:
            gl.glOrtho(-2 * aspect, 2 * aspect, -2, 2, 0.1, 50.0)
        else:
            gl.glOrtho(-2, 2, -2 / aspect, 2 / aspect, 0.1, 50.0)
    gl.glMatrixMode(gl.GL_MODELVIEW)


def draw_axes():
    gl.glPushMatrix()

    gl.glBegin(gl.GL_LINES)

    gl.glColor3f(1.0, 0.0, 0.0)
    gl.glMaterialfv(gl.GL_FRONT, gl.GL_EMISSION, (1.0, 0.0, 0.0, 1.0))
    gl.glVertex3f(0.0, 0.0, 0.0)
    gl.glVertex3f(5.0, 0.0, 0.0)

    gl.glColor3f(0.0, 1.0, 0.0)
    gl.glMaterialfv(gl.GL_FRONT, gl.GL_EMISSION, (0.0, 1.0, 0.0, 1.0))
    gl.glVertex3f(0.0, 0.0, 0.0)
    gl.glVertex3f(0.0, 5.0, 0.0)

    gl.glColor3f(0.0, 0.0, 1.0)
    gl.glMaterialfv(gl.GL_FRONT, gl.GL_EMISSION, (0.0, 0.0, 1.0, 1.0))
    gl.glVertex3f(0.0, 0.0, 0.0)
    gl.glVertex3f(0.0, 0.0, 5.0)

    gl.glEnd()

    gl.glPopMatrix()


def draw(obj: ObjectState):
    gl.glPushMatrix()

    gl.glTranslatef(*obj.translation)
    gl.glScalef(*obj.symmetry)
    gl.glScalef(obj.scale_factor, obj.scale_factor, obj.scale_factor)
    gl.glRotatef(obj.angle[0], 1, 0, 0)
    gl.glRotatef(obj.angle[1], 0, 1, 0)
    gl.glRotatef(obj.angle[2], 0, 0, 1)

    gl.glColor3f(*obj.color)
    gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT, MATERIAL_AMBIENT)
    gl.glMaterialfv(gl.GL_FRONT, gl.GL_DIFFUSE, MATERIAL_DIFFUSE)
    gl.glMaterialfv(gl.GL_FRONT, gl.GL_SPECULAR, MATERIAL_SPECULAR)
    gl.glMaterialfv(gl.GL_FRONT, gl.GL_SHININESS, MATERIAL_SHININESS)
    gl.glMaterialfv(gl.GL_FRONT, gl.GL_EMISSION, (0.0, 0.0, 0.0, 1.0))

    if obj.object_type == "cube":
        glut.glutSolidCube(1.0)
    elif obj.object_type == "sphere":
        glut.glutSolidSphere(0.7, 50, 50)
    elif obj.object_type == "obj" and obj.backing_file_path is not None:
        OBJ(obj.backing_file_path).render()

    if obj.show_axes:
        gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT, (0.0, 0.0, 0.0, 1.0))
        gl.glMaterialfv(gl.GL_FRONT, gl.GL_DIFFUSE, (0.0, 0.0, 0.0, 1.0))
        gl.glMaterialfv(gl.GL_FRONT, gl.GL_SPECULAR, (0.0, 0.0, 0.0, 1.0))
        gl.glMaterialfv(gl.GL_FRONT, gl.GL_SHININESS, (0.0, 0.0, 0.0, 1.0))
        gl.glMaterialfv(gl.GL_FRONT, gl.GL_EMISSION, (0.0, 0.0, 0.0, 1.0))

        gl.glBegin(gl.GL_LINES)

        gl.glColor3f(1.0, 0.0, 0.0)
        gl.glMaterialfv(gl.GL_FRONT, gl.GL_EMISSION, (1.0, 0.0, 0.0, 1.0))
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(1.0, 0.0, 0.0)

        gl.glColor3f(0.0, 1.0, 0.0)
        gl.glMaterialfv(gl.GL_FRONT, gl.GL_EMISSION, (0.0, 1.0, 0.0, 1.0))
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(0.0, 1.0, 0.0)

        gl.glColor3f(0.0, 0.0, 1.0)
        gl.glMaterialfv(gl.GL_FRONT, gl.GL_EMISSION, (0.0, 0.0, 1.0, 1.0))
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(0.0, 0.0, 1.0)

        gl.glEnd()

    gl.glPopMatrix()


def handle_display():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glLoadIdentity()

    glu.gluLookAt(*eye_position, *CENTER_POSITION, *UP_VECTOR)
    draw_axes()
    for old_object in old_objects:
        draw(old_object)
    draw(active_object)

    glut.glutSwapBuffers()


def handle_reshape(width: int, height: int):
    global window_size

    gl.glViewport(0, 0, width, height)
    window_size = (width, height)
    set_projection()


def handle_keyboard(key: bytes, x: int, y: int):
    global hwnd
    global projection
    global active_object
    global edit_mode

    need_redraw = False

    if key == b"x":
        edit_mode = "object" if edit_mode == "camera" else "camera"
        logging.info(f"Edit mode: {edit_mode}")
        return
    elif key == b"\x1b":
        glut.glutDestroyWindow(glut.glutGetWindow())
        return

    if edit_mode == "camera":
        if key == b"p":
            projection = "parallel" if projection == "perspective" else "perspective"
            logging.info(f"Projection: {projection}")
            set_projection()
            need_redraw = True
    elif edit_mode == "object":
        if key == b" ":
            old_objects.append(active_object)
            active_object = ObjectState.initial()
            need_redraw = True
        elif key == b"\x08":
            if len(old_objects) > 0:
                active_object = old_objects.pop()
                need_redraw = True
        elif key == b"q":
            active_object = active_object._replace(
                object_type="sphere" if active_object.object_type == "cube" else "cube"
            )
            need_redraw = True
        elif key == b"r":
            active_object = active_object._replace(
                angle=(0.0, 0.0, 0.0),
                scale_factor=1.0,
                translation=(0.0, 0.0, 0.0),
                symmetry=(1.0, 1.0, 1.0),
            )
            need_redraw = True
        elif key == b"1" or key == b"2" or key == b"3":
            active_object = active_object._replace(
                symmetry=(
                    (
                        -active_object.symmetry[0]
                        if key == b"1"
                        else active_object.symmetry[0]
                    ),
                    (
                        -active_object.symmetry[1]
                        if key == b"2"
                        else active_object.symmetry[1]
                    ),
                    (
                        -active_object.symmetry[2]
                        if key == b"3"
                        else active_object.symmetry[2]
                    ),
                )
            )
            need_redraw = True
        elif key == b"c":
            active_object = active_object._replace(
                color=(
                    random.uniform(0.0, 1.0),
                    random.uniform(0.0, 1.0),
                    random.uniform(0.0, 1.0),
                )
            )
            need_redraw = True
        elif key == b"a":
            active_object = active_object._replace(
                show_axes=not active_object.show_axes
            )
            need_redraw = True
        elif key == b",":
            active_object = active_object._replace(
                angle=(
                    random.uniform(0, 360),
                    random.uniform(0, 360),
                    0.0,
                ),
                color=(
                    random.uniform(0.0, 1.0),
                    random.uniform(0.0, 1.0),
                    random.uniform(0.0, 1.0),
                ),
            )
            old_objects.append(active_object)
            active_object = ObjectState.initial()
            need_redraw = True

    if need_redraw:
        glut.glutPostRedisplay()


def handle_mouse(button: int, button_state: int, x: int, y: int):
    global mouse_lb_down
    global mouse_rb_down
    global mouse_pos
    global active_object
    global eye_position

    need_redraw = False

    mouse_pos = (x, y)

    if button == glut.GLUT_LEFT_BUTTON:
        mouse_lb_down = button_state == glut.GLUT_DOWN
    elif button == glut.GLUT_RIGHT_BUTTON:
        mouse_rb_down = button_state == glut.GLUT_DOWN
    elif button == 3:
        if edit_mode == "camera":
            r, theta, phi = cartesian_to_spherical(*eye_position)
            r += ZOOM_STEP
            eye_position = spherical_to_cartesian(r, theta, phi)
        elif edit_mode == "object":
            active_object = active_object._replace(
                scale_factor=active_object.scale_factor + SCALE_STEP
            )
        need_redraw = True
    elif button == 4:
        if edit_mode == "camera":
            r, theta, phi = cartesian_to_spherical(*eye_position)
            r = max(0, r - ZOOM_STEP)
            eye_position = spherical_to_cartesian(r, theta, phi)
        elif edit_mode == "object":
            active_object = active_object._replace(
                scale_factor=max(0.1, active_object.scale_factor - SCALE_STEP)
            )
        need_redraw = True

    if need_redraw:
        glut.glutPostRedisplay()


def handle_motion(x: int, y: int):
    global mouse_pos
    global active_object
    global eye_position

    need_redraw = False

    dx = x - mouse_pos[0]
    dy = y - mouse_pos[1]
    mouse_pos = (x, y)

    if edit_mode == "object":
        if mouse_rb_down:
            active_object = active_object._replace(
                translation=(
                    active_object.translation[0] + dx * PAN_SENSITIVITY,
                    active_object.translation[1] - dy * PAN_SENSITIVITY,
                    active_object.translation[2],
                )
            )
            need_redraw = True
        if mouse_lb_down:
            active_object = active_object._replace(
                angle=(
                    active_object.angle[0] + dy * ROTATION_SENSITIVITY,
                    active_object.angle[1] + dx * ROTATION_SENSITIVITY,
                    active_object.angle[2],
                )
            )
            need_redraw = True
    elif edit_mode == "camera":
        if mouse_lb_down:
            r, theta, phi = cartesian_to_spherical(*eye_position)

            theta += dx * ROTATION_SENSITIVITY * 0.01
            phi += dy * ROTATION_SENSITIVITY * 0.01
            phi = min(math.pi - 0.001, max(-(math.pi - 0.001), phi))

            eye_position = spherical_to_cartesian(r, theta, phi)

            need_redraw = True

    if need_redraw:
        glut.glutPostRedisplay()


def main():
    logging.basicConfig(level=logging.DEBUG)
    random.seed(0x0D000721)

    glut.glutInit(sys.argv)
    glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA | glut.GLUT_DEPTH)
    glut.glutInitWindowSize(800, 600)
    glut.glutCreateWindow(b"Transformations demo")
    gl.glViewport(0, 0, 800, 600)

    gl.glClearColor(0.0, 0.0, 0.0, 0.0)
    gl.glShadeModel(gl.GL_SMOOTH)
    gl.glEnable(gl.GL_LINE_SMOOTH)
    gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    gl.glLineWidth(2.0)

    gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, LIGHT_POSITION)
    gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, LIGHT_DIFFUSE)
    gl.glLightfv(gl.GL_LIGHT0, gl.GL_SPECULAR, LIGHT_SPECULAR)
    gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, LIGHT_AMBIENT)
    gl.glLightModelfv(gl.GL_LIGHT_MODEL_LOCAL_VIEWER, LIGHT_MODEL_LOCAL_VIEWER)

    gl.glEnable(gl.GL_LIGHTING)
    gl.glEnable(gl.GL_LIGHT0)
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_COLOR_MATERIAL)

    set_projection()

    glut.glutDisplayFunc(handle_display)
    glut.glutReshapeFunc(handle_reshape)
    glut.glutKeyboardFunc(handle_keyboard)
    glut.glutMouseFunc(handle_mouse)
    glut.glutMotionFunc(handle_motion)
    glut.glutMainLoop()


if __name__ == "__main__":
    main()

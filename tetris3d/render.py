import numpy as np
import OpenGL.GL as gl
from const import *
from model import TetrominoShape
from OpenGL.arrays import vbo
from type import *


class BlockRenderer:
    Vertex = tuple[VecXYZf, VecUVf, VecXYZf]  # (pos, yv, normal)

    VERTICES: list[Vertex] = [
        # Front
        ((0.0, 0.0, 1.0), (0.0, 0.0), (0.0, 0.0, 1.0)),
        ((1.0, 0.0, 1.0), (1.0, 0.0), (0.0, 0.0, 1.0)),
        ((1.0, 1.0, 1.0), (1.0, 1.0), (0.0, 0.0, 1.0)),
        ((0.0, 1.0, 1.0), (0.0, 1.0), (0.0, 0.0, 1.0)),
        # Back
        ((0.0, 0.0, 0.0), (1.0, 0.0), (0.0, 0.0, -1.0)),
        ((0.0, 1.0, 0.0), (1.0, 1.0), (0.0, 0.0, -1.0)),
        ((1.0, 1.0, 0.0), (0.0, 1.0), (0.0, 0.0, -1.0)),
        ((1.0, 0.0, 0.0), (0.0, 0.0), (0.0, 0.0, -1.0)),
        # Top
        ((0.0, 1.0, 1.0), (0.0, 1.0), (0.0, 1.0, 0.0)),
        ((1.0, 1.0, 1.0), (1.0, 1.0), (0.0, 1.0, 0.0)),
        ((1.0, 1.0, 0.0), (1.0, 0.0), (0.0, 1.0, 0.0)),
        ((0.0, 1.0, 0.0), (0.0, 0.0), (0.0, 1.0, 0.0)),
        # Bottom
        ((0.0, 0.0, 1.0), (0.0, 0.0), (0.0, -1.0, 0.0)),
        ((0.0, 0.0, 0.0), (0.0, 1.0), (0.0, -1.0, 0.0)),
        ((1.0, 0.0, 0.0), (1.0, 1.0), (0.0, -1.0, 0.0)),
        ((1.0, 0.0, 1.0), (1.0, 0.0), (0.0, -1.0, 0.0)),
        # Right
        ((1.0, 0.0, 1.0), (0.0, 0.0), (1.0, 0.0, 0.0)),
        ((1.0, 0.0, 0.0), (1.0, 0.0), (1.0, 0.0, 0.0)),
        ((1.0, 1.0, 0.0), (1.0, 1.0), (1.0, 0.0, 0.0)),
        ((1.0, 1.0, 1.0), (0.0, 1.0), (1.0, 0.0, 0.0)),
        # Left
        ((0.0, 0.0, 1.0), (1.0, 0.0), (-1.0, 0.0, 0.0)),
        ((0.0, 1.0, 1.0), (1.0, 1.0), (-1.0, 0.0, 0.0)),
        ((0.0, 1.0, 0.0), (0.0, 1.0), (-1.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (0.0, 0.0), (-1.0, 0.0, 0.0)),
    ]

    COLORS: dict[TetrominoShape, tuple[float, float, float]] = {
        TetrominoShape.I: COLOR_CYAN,
        TetrominoShape.O: COLOR_YELLOW,
        TetrominoShape.T: COLOR_PURPLE,
        TetrominoShape.L: COLOR_ORANGE,
        TetrominoShape.J: COLOR_BLUE_LIGHTER,
        TetrominoShape.S: COLOR_GREEN,
        TetrominoShape.Z: COLOR_RED,
    }

    _texture_id: int
    _vertex_vbo: vbo.VBO

    def __init__(self, texture_id: int):
        vertices_data = np.array(
            [
                v
                for vertex in self.VERTICES
                for v in (list(vertex[0]) + list(vertex[1]) + list(vertex[2]))
            ],
            dtype=np.float32,
        )

        self._vertex_vbo = vbo.VBO(vertices_data)
        self._texture_id = texture_id

    def render(self, pos: tuple[float, float, float], type: TetrominoShape):
        gl.glPushMatrix()
        gl.glTranslatef(*pos)

        color = self.COLORS.get(type, COLOR_WHITE)
        gl.glColor3f(*color)
        gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT, MATERIAL_AMBIENT)
        gl.glMaterialfv(gl.GL_FRONT, gl.GL_DIFFUSE, MATERIAL_DIFFUSE)
        gl.glMaterialfv(gl.GL_FRONT, gl.GL_SPECULAR, MATERIAL_SPECULAR)
        gl.glMaterialfv(gl.GL_FRONT, gl.GL_SHININESS, MATERIAL_SHININESS)

        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture_id)
        self._vertex_vbo.bind()
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_TEXTURE_COORD_ARRAY)
        gl.glEnableClientState(gl.GL_NORMAL_ARRAY)

        gl.glVertexPointer(3, gl.GL_FLOAT, 32, self._vertex_vbo)
        gl.glTexCoordPointer(2, gl.GL_FLOAT, 32, self._vertex_vbo + 12)
        gl.glNormalPointer(gl.GL_FLOAT, 32, self._vertex_vbo + 20)
        gl.glDrawArrays(gl.GL_QUADS, 0, len(self.VERTICES))

        gl.glDisableClientState(gl.GL_NORMAL_ARRAY)
        gl.glDisableClientState(gl.GL_TEXTURE_COORD_ARRAY)
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        self._vertex_vbo.unbind()
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glDisable(gl.GL_TEXTURE_2D)

        gl.glPopMatrix()


class MarkerRenderer:
    Vertex = tuple[VecXYZf, ColorRGBf]

    VERTICES: list[Vertex] = [
        ((0.0, 0.0, 0.0), COLOR_RED),
        ((RENDER_BLOCK_SIZE / 3, 0.0, 0.0), COLOR_RED),
        ((0.0, 0.0, 0.0), COLOR_GREEN),
        ((0.0, RENDER_BLOCK_SIZE / 3, 0.0), COLOR_GREEN),
        ((0.0, 0.0, 0.0), COLOR_BLUE),
        ((0.0, 0.0, RENDER_BLOCK_SIZE / 3), COLOR_BLUE),
    ]

    _vertex_vbo: vbo.VBO

    def __init__(self):
        self._vertex_vbo = vbo.VBO(
            np.array(
                [v for vertex in self.VERTICES for v in (list(vertex[0]) + list(vertex[1]))],
                dtype=np.float32,
            )
        )

    def render(self, pos: tuple[float, float, float]):
        gl.glPushMatrix()
        gl.glTranslatef(*pos)

        gl.glLineWidth(1.0)
        gl.glDisable(gl.GL_LIGHTING)

        self._vertex_vbo.bind()
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_COLOR_ARRAY)

        gl.glVertexPointer(3, gl.GL_FLOAT, 24, self._vertex_vbo)
        gl.glColorPointer(3, gl.GL_FLOAT, 24, self._vertex_vbo + 12)
        gl.glDrawArrays(gl.GL_LINES, 0, len(self.VERTICES))

        gl.glDisableClientState(gl.GL_COLOR_ARRAY)
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        self._vertex_vbo.unbind()

        gl.glEnable(gl.GL_LIGHTING)
        gl.glLineWidth(1.0)

        gl.glPopMatrix()


class BorderRenderer:
    Vertex = tuple[VecXYZf, ColorRGBf]

    VERTICES: list[Vertex] = [
        ((0.0, 0.0, 0.0), COLOR_RED),
        ((GAME_AREA_SIZE[0], 0.0, 0.0), COLOR_RED),
        ((GAME_AREA_SIZE[0], 0.0, 0.0), COLOR_WHITE),
        ((GAME_AREA_SIZE[0], 0.0, GAME_AREA_SIZE[1]), COLOR_WHITE),
        ((GAME_AREA_SIZE[0], 0.0, GAME_AREA_SIZE[1]), COLOR_WHITE),
        ((0.0, 0.0, GAME_AREA_SIZE[1]), COLOR_WHITE),
        ((0.0, 0.0, GAME_AREA_SIZE[1]), COLOR_BLUE),
        ((0.0, 0.0, 0.0), COLOR_BLUE),
        ((0.0, 0.0, 0.0), COLOR_GREEN),
        ((0.0, GAME_AREA_SIZE[2], 0.0), COLOR_GREEN),
        ((GAME_AREA_SIZE[0], 0.0, 0.0), COLOR_WHITE),
        ((GAME_AREA_SIZE[0], GAME_AREA_SIZE[2], 0.0), COLOR_WHITE),
        ((GAME_AREA_SIZE[0], 0.0, GAME_AREA_SIZE[1]), COLOR_WHITE),
        ((GAME_AREA_SIZE[0], GAME_AREA_SIZE[2], GAME_AREA_SIZE[1]), COLOR_WHITE),
        ((0.0, 0.0, GAME_AREA_SIZE[1]), COLOR_WHITE),
        ((0.0, GAME_AREA_SIZE[2], GAME_AREA_SIZE[1]), COLOR_WHITE),
        ((0.0, GAME_AREA_SIZE[2], 0.0), COLOR_WHITE),
        ((GAME_AREA_SIZE[0], GAME_AREA_SIZE[2], 0.0), COLOR_WHITE),
        ((GAME_AREA_SIZE[0], GAME_AREA_SIZE[2], 0.0), COLOR_WHITE),
        ((GAME_AREA_SIZE[0], GAME_AREA_SIZE[2], GAME_AREA_SIZE[1]), COLOR_WHITE),
        ((GAME_AREA_SIZE[0], GAME_AREA_SIZE[2], GAME_AREA_SIZE[1]), COLOR_WHITE),
        ((0.0, GAME_AREA_SIZE[2], GAME_AREA_SIZE[1]), COLOR_WHITE),
        ((0.0, GAME_AREA_SIZE[2], GAME_AREA_SIZE[1]), COLOR_WHITE),
        ((0.0, GAME_AREA_SIZE[2], 0.0), COLOR_WHITE),
    ]

    _vertex_vbo: vbo.VBO

    def __init__(self):
        self._vertex_vbo = vbo.VBO(
            np.array(
                [v for vertex in self.VERTICES for v in (list(vertex[0]) + list(vertex[1]))],
                dtype=np.float32,
            )
        )

    def render(self):
        gl.glPushMatrix()

        gl.glLineWidth(2.0)
        gl.glDisable(gl.GL_LIGHTING)

        self._vertex_vbo.bind()
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_COLOR_ARRAY)

        gl.glVertexPointer(3, gl.GL_FLOAT, 24, self._vertex_vbo)
        gl.glColorPointer(3, gl.GL_FLOAT, 24, self._vertex_vbo + 12)
        gl.glDrawArrays(gl.GL_LINES, 0, len(self.VERTICES))

        gl.glDisableClientState(gl.GL_COLOR_ARRAY)
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        self._vertex_vbo.unbind()

        gl.glEnable(gl.GL_LIGHTING)
        gl.glLineWidth(1.0)

        gl.glPopMatrix()

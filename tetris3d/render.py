import numpy as np
import OpenGL.GL as gl
from const import *
from model import TetrominoShape
from OpenGL.arrays import vbo


class BlockRenderer:
    VertexXYZ = tuple[float, float, float]
    VertexUV = tuple[float, float]
    VertexNormal = tuple[float, float, float]
    Vertex = tuple[VertexXYZ, VertexUV, VertexNormal]

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

    texture_id: int
    vertex_vbo: vbo.VBO

    def __init__(self, texture_id: int):
        vertices_data = np.array(
            [
                v
                for vertex in self.VERTICES
                for v in (list(vertex[0]) + list(vertex[1]) + list(vertex[2]))
            ],
            dtype=np.float32,
        )

        self.vertex_vbo = vbo.VBO(vertices_data)
        self.texture_id = texture_id

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
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        self.vertex_vbo.bind()
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_TEXTURE_COORD_ARRAY)
        gl.glEnableClientState(gl.GL_NORMAL_ARRAY)

        gl.glVertexPointer(3, gl.GL_FLOAT, 32, self.vertex_vbo)
        gl.glTexCoordPointer(2, gl.GL_FLOAT, 32, self.vertex_vbo + 12)
        gl.glNormalPointer(gl.GL_FLOAT, 32, self.vertex_vbo + 20)
        gl.glDrawArrays(gl.GL_QUADS, 0, 24)

        gl.glDisableClientState(gl.GL_NORMAL_ARRAY)
        gl.glDisableClientState(gl.GL_TEXTURE_COORD_ARRAY)
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        self.vertex_vbo.unbind()
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glDisable(gl.GL_TEXTURE_2D)

        gl.glPopMatrix()


class MarkerRenderer:
    VertexXYZ = tuple[float, float, float]
    VertexColor = tuple[float, float, float]
    Vertex = tuple[VertexXYZ, VertexColor]

    VERTICES: list[Vertex] = [
        ((0.0, 0.0, 0.0), COLOR_RED),
        ((RENDER_BLOCK_SIZE / 3, 0.0, 0.0), COLOR_RED),
        ((0.0, 0.0, 0.0), COLOR_GREEN),
        ((0.0, RENDER_BLOCK_SIZE / 3, 0.0), COLOR_GREEN),
        ((0.0, 0.0, 0.0), COLOR_BLUE),
        ((0.0, 0.0, RENDER_BLOCK_SIZE / 3), COLOR_BLUE),
    ]

    vertex_vbo: vbo.VBO

    def __init__(self):
        self.vertex_vbo = vbo.VBO(
            np.array(
                [v for vertex in self.VERTICES for v in (list(vertex[0]) + list(vertex[1]))],
                dtype=np.float32,
            )
        )

    def render(self, pos: tuple[float, float, float]):
        gl.glPushMatrix()
        gl.glTranslatef(*pos)

        gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT, MATERIAL_NULL)
        gl.glMaterialfv(gl.GL_FRONT, gl.GL_DIFFUSE, MATERIAL_NULL)
        gl.glMaterialfv(gl.GL_FRONT, gl.GL_SPECULAR, MATERIAL_NULL)
        gl.glMaterialfv(gl.GL_FRONT, gl.GL_SHININESS, MATERIAL_SHININESS)

        self.vertex_vbo.bind()
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_COLOR_ARRAY)

        gl.glVertexPointer(3, gl.GL_FLOAT, 24, self.vertex_vbo)
        gl.glColorPointer(3, gl.GL_FLOAT, 24, self.vertex_vbo + 12)
        gl.glDrawArrays(gl.GL_LINES, 0, 6)

        gl.glDisableClientState(gl.GL_COLOR_ARRAY)
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        self.vertex_vbo.unbind()

        gl.glPopMatrix()

import math
import bpy
import os
import bmesh
from contextlib import contextmanager
from typing import List
from math import sqrt, acos, pow
from mathutils import Vector, Matrix, Euler
from bpy.types import Menu
from bpy.props import FloatProperty, BoolProperty, StringProperty, EnumProperty, PointerProperty, IntProperty
from bpy_extras.object_utils import object_data_add

bl_info = {
    'name': 'DiceGen 5.x',
    'author': 'Long Tran, shawn-makes-stuff',
    'version': (1, 2, 0),
    'blender': (5, 0, 0),
    'location': 'View3D > Add > Mesh',
    'description': 'Generate polyhedral dice models.',
    'category': 'Add Mesh',
    'doc_url': 'https://github.com/Longi94/blender-dice-gen/wiki',
    'tracker_url': 'https://github.com/Longi94/blender-dice-gen/issues'
}

NUMBER_IND_NONE = 'none'
NUMBER_IND_BAR = 'bar'
NUMBER_IND_PERIOD = 'period'

HALF_PI = math.pi / 2


def leg_b(leg_a, h):
    """given a leg of a right angle triangle and it's height, calculate the other leg"""
    return sqrt(pow(h, 2) + (pow(h, 4) / (pow(leg_a, 2) - pow(h, 2))))


# https://dmccooey.com/polyhedra
CONSTANTS = {
    'tetrahedron': {
        'dihedral_angle': acos(1 / 3),
        'height': sqrt(2 / 3),
        'c0': sqrt(2) / 4
    },
    'octahedron': {
        'dihedral_angle': acos(sqrt(5) / -5),
        'circumscribed_r': (sqrt(3) + sqrt(15)) / 4,
        'inscribed_r': sqrt(10 * (25 + 11 * sqrt(5))) / 20,
        'c0': (1 + sqrt(5)) / 4,
        'c1': (3 + sqrt(5)) / 4,
        'c2': 0.5
    },
    'icosahedron': {
        'dihedral_angle': acos(sqrt(5) / -3),
        'circumscribed_r': sqrt(2 * (5 + sqrt(5))) / 4,
        'inscribed_r': (3 * sqrt(3) + sqrt(15)) / 12,
        'c0': (1 + sqrt(5)) / 4,
        'c1': 0.5
    },
    'pentagonal_trap': {
        'inscribed_r': sqrt(5 * (5 + 2 * sqrt(5))) / 10,
        'base_height': 1.1180340051651,
        'base_width': leg_b(1.1180340051651, 0.5),
        'c0': (sqrt(5) - 1) / 4,
        'c1': (1 + sqrt(5)) / 4,
        'c2': (3 + sqrt(5)) / 4,
        'c3': 0.5
    }
}

# calculate rotation of trapezohedron to have it stand upright
# from dice-gen Math.acos((C0 - C2) / Math.sqrt(Math.pow(C0 - C2, 2) + 4 * Math.pow(C1, 2)))
CONSTANTS['pentagonal_trap']['angle'] = Euler((0, 0, acos(
    (CONSTANTS['pentagonal_trap']['c0'] - CONSTANTS['pentagonal_trap']['c2']) / sqrt(
        pow(CONSTANTS['pentagonal_trap']['c0'] - CONSTANTS['pentagonal_trap']['c2'], 2) + 4 * pow(
            CONSTANTS['pentagonal_trap']['c1'], 2)))), 'XYZ')

CONSTANTS['pentagonal_trap']['angle'].rotate(Euler((HALF_PI, 0, 0), 'XYZ'))


class Mesh:

    def __init__(self, name):
        self.vertices = None
        self.faces = None
        self.name = name
        self.dice_mesh = None
        self.base_font_scale = 1

    def create(self, context):
        self.dice_mesh = create_mesh(context, self.vertices, self.faces, self.name)
        # reset transforms
        self.dice_mesh.matrix_world = Matrix()
        return self.dice_mesh

    def get_numbers(self):
        return []

    def get_number_locations(self):
        return []

    def get_number_rotations(self):
        return []

    def create_numbers(self, context, size, number_scale, number_depth, font_path, one_offset,
                       number_indicator_type=NUMBER_IND_NONE, period_indicator_scale=1, period_indicator_space=1,
                       bar_indicator_height=1, bar_indicator_width=1, bar_indicator_space=1,
                       center_bar=True, custom_image_face=0, custom_image_path='', custom_image_scale=1):
        numbers = self.get_numbers()
        locations = self.get_number_locations()
        rotations = self.get_number_rotations()

        font_size = self.base_font_scale * size * number_scale

        numbers_object = create_numbers(context, numbers, locations, rotations, font_path, font_size, number_depth,
                                        number_indicator_type, period_indicator_scale, period_indicator_space,
                                        bar_indicator_height, bar_indicator_width, bar_indicator_space,
                                        center_bar, one_offset, custom_image_face=custom_image_face,
                                        custom_image_path=custom_image_path, custom_image_scale=custom_image_scale)

        if numbers_object is not None:
            numbers_object.name = "dice_numbers"
            apply_boolean_modifier(self.dice_mesh, numbers_object)
            return numbers_object

        return None


class Tetrahedron(Mesh):
    def __init__(self, name, size, number_center_offset):
        super().__init__(name)
        self.size = size
        self.number_center_offset = number_center_offset

        c0 = CONSTANTS['tetrahedron']['c0'] / CONSTANTS['tetrahedron']['height'] * size

        self.vertices = [(c0, -c0, c0), (c0, c0, -c0), (-c0, c0, c0), (-c0, -c0, -c0)]
        self.faces = [[0, 1, 2], [1, 0, 3], [2, 3, 0], [3, 2, 1]]

        self.base_font_scale = 0.3

    def get_numbers(self):
        return [str(math.floor(i / 3) + 1) for i in range(12)]

    def get_number_locations(self):
        # face centers
        centers = [Vector((
            ((self.vertices[f[0]][0] + self.vertices[f[1]][0] + self.vertices[f[2]][0]) / 3),
            ((self.vertices[f[0]][1] + self.vertices[f[1]][1] + self.vertices[f[2]][1]) / 3),
            ((self.vertices[f[0]][2] + self.vertices[f[1]][2] + self.vertices[f[2]][2]) / 3)
        )) for f in self.faces]
        vertices = [Vector(v) for v in self.vertices]

        location_vectors = [
            centers[0].lerp(vertices[2], self.number_center_offset),
            centers[2].lerp(vertices[2], self.number_center_offset),
            centers[3].lerp(vertices[2], self.number_center_offset),
            centers[0].lerp(vertices[1], self.number_center_offset),
            centers[1].lerp(vertices[1], self.number_center_offset),
            centers[3].lerp(vertices[1], self.number_center_offset),
            centers[0].lerp(vertices[0], self.number_center_offset),
            centers[1].lerp(vertices[0], self.number_center_offset),
            centers[2].lerp(vertices[0], self.number_center_offset),
            centers[1].lerp(vertices[3], self.number_center_offset),
            centers[2].lerp(vertices[3], self.number_center_offset),
            centers[3].lerp(vertices[3], self.number_center_offset)
        ]

        return [(v.x, v.y, v.z) for v in location_vectors]

    def get_number_rotations(self):
        return [
            (CONSTANTS['tetrahedron']['dihedral_angle'] / 2, math.pi / 4, HALF_PI),
            (CONSTANTS['tetrahedron']['dihedral_angle'] / 2, -math.pi / 4, 0),
            (-(math.pi - CONSTANTS['tetrahedron']['dihedral_angle']) / 2, math.pi, math.pi / 4),
            (-(math.pi - CONSTANTS['tetrahedron']['dihedral_angle']) / 2, 0, -math.pi / 4),
            (CONSTANTS['tetrahedron']['dihedral_angle'] / 2, math.pi * 3 / 4, 0),
            (CONSTANTS['tetrahedron']['dihedral_angle'] / 2, math.pi * 5 / 4, math.pi * 3 / 2),
            (CONSTANTS['tetrahedron']['dihedral_angle'] / 2, -math.pi / 4, -math.pi),
            (-(math.pi - CONSTANTS['tetrahedron']['dihedral_angle']) / 2, math.pi, -math.pi * 3 / 4),
            (CONSTANTS['tetrahedron']['dihedral_angle'] / 2, math.pi / 4, -HALF_PI),
            (CONSTANTS['tetrahedron']['dihedral_angle'] / 2, math.pi * 5 / 4, HALF_PI),
            (-(math.pi - CONSTANTS['tetrahedron']['dihedral_angle']) / 2, 0, math.pi * 3 / 4),
            (CONSTANTS['tetrahedron']['dihedral_angle'] / 2, math.pi * 3 / 4, math.pi)
        ]


class D4Crystal(Mesh):

    def __init__(self, name, size, base_height, point_height):
        super().__init__(name)
        self.size = size

        c0 = 0.5 * size
        c1 = 0.5 * base_height
        c2 = 0.5 * base_height + point_height

        self.vertices = [(-c0, -c0, c1), (c0, -c0, c1), (c0, c0, c1), (-c0, c0, c1), (-c0, -c0, -c1), (c0, -c0, -c1),
                         (c0, c0, -c1), (-c0, c0, -c1), (0, 0, c2), (0, 0, -c2)]
        self.faces = [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [0, 1, 8], [1, 2, 8], [2, 3, 8],
                      [3, 0, 8], [4, 5, 9], [5, 6, 9], [6, 7, 9], [7, 4, 9]]

        self.base_font_scale = 0.8

    def get_numbers(self):
        return numbers(4)

    def get_number_locations(self):
        c0 = 0.5 * self.size
        return [(c0, 0, 0), (0, c0, 0), (0, -c0, 0), (-c0, 0, 0)]

    def get_number_rotations(self):
        return [(HALF_PI, 0, HALF_PI), (HALF_PI, 0, HALF_PI * 2), (HALF_PI, 0, 0), (HALF_PI, 0, HALF_PI * 3)]


class D4Shard(Mesh):

    def __init__(self, name, size, top_point_height, bottom_point_height, number_v_offset):
        super().__init__(name)
        self.size = size
        self.number_v_offset = number_v_offset
        self.bottom_point_height = bottom_point_height

        c0 = size / sqrt(2)
        c1 = top_point_height * c0
        c2 = bottom_point_height * c0

        self.vertices = [(c0, 0, 0), (0, c0, 0), (0, -c0, 0), (-c0, 0, 0), (0, 0, c1), (0, 0, -c2)]
        self.faces = [[0, 1, 4], [1, 3, 4], [3, 2, 4], [2, 0, 4], [0, 1, 5], [1, 3, 5], [3, 2, 5], [2, 0, 5]]

        self.base_font_scale = 0.8

    def get_numbers(self):
        return numbers(4)

    def get_number_locations(self):
        c0 = self.size / 2 / sqrt(2) * self.number_v_offset
        c1 = self.size / sqrt(2) * self.bottom_point_height * (1 - self.number_v_offset)
        return [(c0, c0, -c1), (-c0, c0, -c1), (c0, -c0, -c1), (-c0, -c0, -c1)]

    def get_number_rotations(self):
        c0 = self.size / 2 / sqrt(2)
        c1 = self.size / sqrt(2) * self.bottom_point_height
        angle = math.pi / 2 + Vector((0, 0, c1)).angle(Vector((c0, c0, c1)))
        return [(angle, 0, math.pi * 3 / 4), (angle, 0, math.pi * 5 / 4), (angle, 0, math.pi * 1 / 4),
                (angle, 0, math.pi * 7 / 4)]


class Cube(Mesh):

    def __init__(self, name, size):
        super().__init__(name)

        # Calculate the necessary constants
        self.v_coord_const = 0.5 * size
        s = self.v_coord_const

        # create the vertices and faces
        self.vertices = [(-s, -s, -s), (s, -s, -s), (s, s, -s), (-s, s, -s), (-s, -s, s), (s, -s, s), (s, s, s),
                         (-s, s, s)]
        self.faces = [[0, 3, 2, 1], [0, 1, 5, 4], [0, 4, 7, 3], [6, 5, 1, 2], [6, 2, 3, 7], [6, 7, 4, 5]]

    def get_numbers(self):
        return numbers(6)

    def get_number_locations(self):
        s = self.v_coord_const
        return [(0, -s, 0), (-s, 0, 0), (0, 0, s), (0, 0, -s), (s, 0, 0), (0, s, 0)]

    def get_number_rotations(self):
        return [
            (HALF_PI, 0, 0),
            (math.pi, HALF_PI, 0),
            (0, 0, 0),
            (math.pi, 0, 0),
            (0, HALF_PI, 0),
            (-HALF_PI, 0, 0)
        ]


class Octahedron(Mesh):

    def __init__(self, name, size):
        super().__init__(name)

        # calculate circumscribed sphere radius from inscribed sphere radius
        # diameter of the inscribed sphere is the face 2 face length of the octahedron
        self.circumscribed_r = (size * math.sqrt(3)) / 2
        s = self.circumscribed_r

        # create the vertices and faces
        self.vertices = [(s, 0, 0), (-s, 0, 0), (0, s, 0), (0, -s, 0), (0, 0, s), (0, 0, -s)]
        self.faces = [[4, 0, 2], [4, 2, 1], [4, 1, 3], [4, 3, 0], [5, 2, 0], [5, 1, 2], [5, 3, 1], [5, 0, 3]]

        self.base_font_scale = 0.7

    def get_numbers(self):
        return numbers(8)

    def get_number_locations(self):
        c = self.circumscribed_r / 3
        return [
            (c, c, c),
            (c, c, -c),
            (c, -c, -c),
            (c, -c, c),
            (-c, c, -c),
            (-c, c, c),
            (-c, -c, c),
            (-c, -c, -c),
        ]

    def get_number_rotations(self):
        # dihedral angle / 2
        da = math.acos(-1 / 3)
        return [
            (da / 2, 0, math.pi * 3 / 4),
            (-math.pi + da / 2, 0, -math.pi / 4),
            (-math.pi + da / 2, 0, -math.pi * 3 / 4),
            (da / 2, 0, math.pi * 1 / 4),
            (-math.pi + da / 2, 0, math.pi * 1 / 4),
            (da / 2, 0, -math.pi * 3 / 4),
            (da / 2, 0, -math.pi * 1 / 4),
            (-math.pi + da / 2, 0, math.pi * 3 / 4),
        ]


class Dodecahedron(Mesh):

    def __init__(self, name, size):
        super().__init__(name)
        self.size = size

        # Calculate the necessary constants https://dmccooey.com/polyhedra/Dodecahedron.html
        edge_length = size / 2 / CONSTANTS['octahedron']['inscribed_r']

        c0 = CONSTANTS['octahedron']['c0'] * edge_length
        c1 = CONSTANTS['octahedron']['c1'] * edge_length
        s = CONSTANTS['octahedron']['c2'] * edge_length

        self.vertices = [(0.0, s, c1), (0.0, s, -c1), (0.0, -s, c1), (0.0, -s, -c1), (c1, 0.0, s), (c1, 0.0, -s),
                         (-c1, 0.0, s), (-c1, 0.0, -s), (s, c1, 0.0), (s, -c1, 0.0), (-s, c1, 0.0), (-s, -c1, 0.0),
                         (c0, c0, c0), (c0, c0, -c0), (c0, -c0, c0), (c0, -c0, -c0), (-c0, c0, c0), (-c0, c0, -c0),
                         (-c0, -c0, c0), (-c0, -c0, -c0)]

        self.faces = [[0, 2, 14, 4, 12], [0, 12, 8, 10, 16], [0, 16, 6, 18, 2], [7, 6, 16, 10, 17],
                      [7, 17, 1, 3, 19], [7, 19, 11, 18, 6], [9, 11, 19, 3, 15], [9, 15, 5, 4, 14],
                      [9, 14, 2, 18, 11], [13, 1, 17, 10, 8], [13, 8, 12, 4, 5], [13, 5, 15, 3, 1]]

        self.base_font_scale = 0.5

    def get_numbers(self):
        return numbers(12)

    def get_number_locations(self):
        dual_e = self.size / 2 / CONSTANTS['icosahedron']['circumscribed_r']
        c0 = dual_e * CONSTANTS['icosahedron']['c0']
        c1 = dual_e * CONSTANTS['icosahedron']['c1']
        return [
            (c1, 0.0, c0),
            (0.0, c0, c1),
            (-c1, 0.0, c0),
            (0.0, -c0, c1),
            (c0, -c1, 0.0),
            (c0, c1, 0.0),
            (-c0, -c1, 0.0),
            (-c0, c1, 0.0),
            (0.0, c0, -c1),
            (c1, 0.0, -c0),
            (0.0, -c0, -c1),
            (-c1, 0.0, -c0),
        ]

    def get_number_rotations(self):
        angles = [Euler((0, 0, 0), 'XYZ') for _ in range(12)]

        angles[0].z = math.radians(-162)
        angles[0].rotate(Euler((0, (math.pi - CONSTANTS['octahedron']['dihedral_angle']) / 2, 0), 'XYZ'))

        angles[1].z = math.radians(36)
        angles[1].rotate(Euler((CONSTANTS['octahedron']['dihedral_angle'] / -2, 0, 0), 'XYZ'))

        angles[2].z = HALF_PI
        angles[2].x = -(math.pi - CONSTANTS['octahedron']['dihedral_angle']) / 2

        angles[3].z = math.radians(144)
        angles[3].rotate(Euler((CONSTANTS['octahedron']['dihedral_angle'] / 2, 0, 0), 'XYZ'))

        angles[4].y = HALF_PI
        angles[4].rotate(Euler((-math.radians(108), 0, 0), 'XYZ'))
        angles[4].rotate(Euler((0, 0, (math.pi - CONSTANTS['octahedron']['dihedral_angle']) / -2), 'XYZ'))

        angles[5].y = HALF_PI
        angles[5].rotate(Euler((-math.radians(72), 0, 0), 'XYZ'))
        angles[5].rotate(Euler((0, 0, (math.pi - CONSTANTS['octahedron']['dihedral_angle']) / 2), 'XYZ'))

        angles[6].y = -HALF_PI
        angles[6].rotate(Euler((math.radians(108), 0, 0), 'XYZ'))
        angles[6].rotate(Euler((0, 0, (math.pi - CONSTANTS['octahedron']['dihedral_angle']) / 2), 'XYZ'))

        angles[7].y = HALF_PI
        angles[7].rotate(Euler((math.radians(72), 0, 0), 'XYZ'))
        angles[7].rotate(Euler((0, 0, -(math.pi - CONSTANTS['octahedron']['dihedral_angle']) / 2), 'XYZ'))

        angles[8].z = math.radians(-36)
        angles[8].y = math.pi
        angles[8].rotate(Euler((CONSTANTS['octahedron']['dihedral_angle'] / 2, 0, 0), 'XYZ'))

        angles[9].x = math.pi
        angles[9].z = HALF_PI
        angles[9].rotate(Euler((0, -(math.pi - CONSTANTS['octahedron']['dihedral_angle']) / 2, 0), 'XYZ'))

        angles[10].x = math.pi
        angles[10].z = math.radians(36)
        angles[10].rotate(Euler((-CONSTANTS['octahedron']['dihedral_angle'] / 2, 0, 0), 'XYZ'))

        angles[11].z = math.radians(342)
        angles[11].rotate(Euler((0, math.pi + (math.pi - CONSTANTS['octahedron']['dihedral_angle']) / 2, 0), 'XYZ'))

        return [(a.x, a.y, a.z) for a in angles]


class Icosahedron(Mesh):

    def __init__(self, name, size):
        super().__init__(name)
        self.size = size

        # Calculate the necessary constants https://dmccooey.com/polyhedra/Icosahedron.html
        edge_length = size / 2 / CONSTANTS['icosahedron']['inscribed_r']

        c0 = edge_length * CONSTANTS['icosahedron']['c0']
        c1 = edge_length * CONSTANTS['icosahedron']['c1']

        self.vertices = [(c1, 0.0, c0), (c1, 0.0, -c0), (-c1, 0.0, c0), (-c1, 0.0, -c0), (c0, c1, 0.0), (c0, -c1, 0.0),
                         (-c0, c1, 0.0), (-c0, -c1, 0.0), (0.0, c0, c1), (0.0, c0, -c1), (0.0, -c0, c1),
                         (0.0, -c0, -c1)]
        self.faces = [[0, 2, 10], [0, 10, 5], [0, 5, 4], [0, 4, 8], [0, 8, 2], [3, 1, 11], [3, 11, 7], [3, 7, 6],
                      [3, 6, 9], [3, 9, 1], [2, 6, 7], [2, 7, 10], [10, 7, 11], [10, 11, 5], [5, 11, 1], [5, 1, 4],
                      [4, 1, 9], [4, 9, 8], [8, 9, 6], [8, 6, 2]]

        self.base_font_scale = 0.3

    def get_numbers(self):
        return numbers(20)

    def get_number_locations(self):
        dual_e = self.size / 2 / CONSTANTS['octahedron']['circumscribed_r']

        c0 = CONSTANTS['octahedron']['c0'] * dual_e
        c1 = CONSTANTS['octahedron']['c1'] * dual_e
        s = CONSTANTS['octahedron']['c2'] * dual_e

        return [
            (0.0, s, c1),
            (-c0, -c0, -c0),
            (s, c1, 0.0),
            (s, -c1, 0.0),
            (-c0, -c0, c0),
            (c1, 0.0, -s),
            (-c0, c0, c0),
            (0.0, s, -c1),
            (c1, 0.0, s),
            (-c0, c0, -c0),
            (c0, -c0, c0),
            (-c1, 0.0, -s),
            (0.0, -s, c1),
            (c0, -c0, -c0),
            (-c1, 0.0, s),
            (c0, c0, -c0),
            (-s, c1, 0.0),
            (-s, -c1, 0.0),
            (c0, c0, c0),
            (0.0, -s, -c1)
        ]

    def get_number_rotations(self):
        angles = [Euler((0, 0, 0), 'XYZ') for _ in range(20)]

        # TODO magic numbers copied out of blender with alignment trick, try to find exact equation

        angles[0].x = -(math.pi - CONSTANTS['icosahedron']['dihedral_angle']) / 2

        angles[1].x = -0.918438
        angles[1].y = -2.82743
        angles[1].z = -4.15881

        angles[2].x = HALF_PI
        angles[2].y = 5 / 6 * math.pi
        angles[2].z = math.pi - ((math.pi - CONSTANTS['icosahedron']['dihedral_angle']) / 2)

        angles[3].x = HALF_PI
        angles[3].y = -math.pi / 6
        angles[3].z = (math.pi - CONSTANTS['icosahedron']['dihedral_angle']) / 2

        angles[4].x = -0.918438
        angles[4].y = -0.314159
        angles[4].z = 2.12437

        angles[5].x = HALF_PI
        angles[5].y = math.pi / 3
        angles[5].z = HALF_PI
        angles[5].rotate(Euler((0, (math.pi - CONSTANTS['icosahedron']['dihedral_angle']) / 2, 0), 'XYZ'))

        angles[6].x = -0.918438
        angles[6].y = 0.314159
        angles[6].z = 1.01722

        angles[7].x = -(math.pi - CONSTANTS['icosahedron']['dihedral_angle']) / 2
        angles[7].y = math.pi

        angles[8].x = -HALF_PI
        angles[8].y = -math.pi / 3
        angles[8].z = -HALF_PI
        angles[8].rotate(Euler((0, -(math.pi - CONSTANTS['icosahedron']['dihedral_angle']) / 2, 0), 'XYZ'))

        angles[9].x = -4.06003
        angles[9].y = 0.314159
        angles[9].z = -2.12437

        angles[10].x = -0.918438
        angles[10].y = 0.314159
        angles[10].z = -2.12437

        angles[11].x = HALF_PI
        angles[11].y = -math.pi / 3
        angles[11].z = -HALF_PI
        angles[11].rotate(Euler((0, -(math.pi - CONSTANTS['icosahedron']['dihedral_angle']) / 2, 0), 'XYZ'))

        angles[12].x = -(math.pi - CONSTANTS['icosahedron']['dihedral_angle']) / 2
        angles[12].z = math.pi

        angles[13].x = 2.22315
        angles[13].y = 0.314159
        angles[13].z = 1.01722

        angles[14].x = -HALF_PI
        angles[14].y = math.pi / 3
        angles[14].z = HALF_PI
        angles[14].rotate(Euler((0, (math.pi - CONSTANTS['icosahedron']['dihedral_angle']) / 2, 0), 'XYZ'))

        angles[15].x = -0.918438
        angles[15].y = -2.82743
        angles[15].z = -1.01722

        angles[16].x = HALF_PI
        angles[16].y = 7 / 6 * math.pi
        angles[16].z = math.pi + ((math.pi - CONSTANTS['icosahedron']['dihedral_angle']) / 2)

        angles[17].x = HALF_PI
        angles[17].y = math.pi / 6
        angles[17].z = -(math.pi - CONSTANTS['icosahedron']['dihedral_angle']) / 2

        angles[18].x = -0.918438
        angles[18].y = -0.314159
        angles[18].z = -1.01722

        angles[19].x = math.pi
        angles[19].z = 2 / 3 * math.pi
        angles[19].rotate(Euler((-(math.pi - CONSTANTS['icosahedron']['dihedral_angle']) / 2, 0, 0), 'XYZ'))

        return [(a.x, a.y, a.z) for a in angles]


class SquashedPentagonalTrapezohedron(Mesh):

    def __init__(self, name, size, height, number_v_offset):
        super().__init__(name)
        self.size = size
        self.height = height
        self.number_v_offset = number_v_offset

        antiprism_e = size / 2 / CONSTANTS['pentagonal_trap']['inscribed_r']

        c0 = CONSTANTS['pentagonal_trap']['c0'] * antiprism_e
        c1 = CONSTANTS['pentagonal_trap']['c1'] * antiprism_e
        c2 = CONSTANTS['pentagonal_trap']['c2'] * antiprism_e
        c3 = CONSTANTS['pentagonal_trap']['c3'] * antiprism_e

        scaled_base_height = CONSTANTS['pentagonal_trap']['base_height'] * size
        scaled_base_width = CONSTANTS['pentagonal_trap']['base_width'] * size

        scaled_height = scaled_base_height * height
        scaled_width = leg_b(scaled_height, size / 2)
        width = scaled_width / scaled_base_width

        # TODO figure out where this angle comes from
        self.vertices = [(0.0, c0, c1), (0.0, c0, -c1), (0.0, -c0, c1), (0.0, -c0, -c1), (c3, c3, c3), (c3, c3, -c3),
                         (-c3, -c3, c3), (-c3, -c3, -c3), (c2, -c1, 0.0), (-c2, c1, 0.0), (c0, c1, 0.0),
                         (-c0, -c1, 0.0)]

        def transform(v):
            # rotate the vectors, so the trapezohedron is up right
            vector = Vector(v)
            vector.rotate(CONSTANTS['pentagonal_trap']['angle'])

            # scale the body
            vector.z *= height
            vector.y *= width
            vector.x *= width
            return vector.x, vector.y, vector.z

        self.vertices = list(map(transform, self.vertices))

        self.faces = [[8, 2, 6, 11], [8, 11, 7, 3], [8, 3, 1, 5], [8, 5, 10, 4], [8, 4, 0, 2], [9, 0, 4, 10],
                      [9, 10, 5, 1], [9, 1, 3, 7], [9, 7, 11, 6], [9, 6, 2, 0]]

    def get_number_locations(self):
        vectors = [Vector(v) for v in self.vertices]

        lerp_factor = self.number_v_offset
        location_vectors = [
            vectors[6].lerp(vectors[8], lerp_factor),
            vectors[3].lerp(vectors[9], lerp_factor),
            vectors[1].lerp(vectors[8], lerp_factor),
            vectors[4].lerp(vectors[9], lerp_factor),
            vectors[10].lerp(vectors[8], lerp_factor),
            vectors[11].lerp(vectors[9], lerp_factor),
            vectors[7].lerp(vectors[8], lerp_factor),
            vectors[2].lerp(vectors[9], lerp_factor),
            vectors[0].lerp(vectors[8], lerp_factor),
            vectors[5].lerp(vectors[9], lerp_factor)
        ]

        return [(v.x, v.y, v.z) for v in location_vectors]

    def get_number_rotations(self):
        a = Vector(self.vertices[9])
        b = Vector(self.vertices[10]) - Vector(self.vertices[8])
        number_angle = HALF_PI - a.angle(b)
        return [
            (number_angle, 0, -HALF_PI - math.pi * 6 / 5),
            (math.pi + number_angle, 0, -HALF_PI - math.pi * 8 / 5),
            (number_angle, 0, -HALF_PI - math.pi * 2 / 5),
            (math.pi + number_angle, 0, -HALF_PI - math.pi * 4 / 5),
            (number_angle, 0, -HALF_PI),
            (math.pi + number_angle, 0, -HALF_PI),
            (number_angle, 0, -HALF_PI - math.pi * 4 / 5),
            (math.pi + number_angle, 0, -HALF_PI - math.pi * 2 / 5),
            (number_angle, 0, -HALF_PI - math.pi * 8 / 5),
            (math.pi + number_angle, 0, -HALF_PI - math.pi * 6 / 5)
        ]


class D10Mesh(SquashedPentagonalTrapezohedron):

    def __init__(self, name, size, height, number_v_offset):
        super().__init__(name, size, height, number_v_offset)
        self.base_font_scale = 0.6

    def get_numbers(self):
        return [str((i + 1) % 10) for i in range(10)]


class D100Mesh(SquashedPentagonalTrapezohedron):

    def __init__(self, name, size, height, number_v_offset):
        super().__init__(name, size, height, number_v_offset)
        self.base_font_scale = 0.45

    def get_numbers(self):
        return [f'{str((i + 1) % 10)}0' for i in range(10)]


def numbers(n: int) -> List[str]:
    return [str(i + 1) for i in range(n)]


def set_origin(o, v):
    """
    set origin to a specific location
    """
    me = o.data
    mw = o.matrix_world
    current = o.location
    T = Matrix.Translation(current - v)
    me.transform(T)
    mw.translation = mw @ v


def _calculate_bounds(vertices):
    iterator = iter(vertices)
    try:
        first_vertex = next(iterator)
    except StopIteration:
        return None

    min_x = max_x = first_vertex.co.x
    min_y = max_y = first_vertex.co.y
    min_z = max_z = first_vertex.co.z

    for v in iterator:
        x, y, z = v.co.x, v.co.y, v.co.z
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)
        min_z = min(min_z, z)
        max_z = max(max_z, z)

    return min_x, max_x, min_y, max_y, min_z, max_z


def set_origin_center_bounds(o):
    """
    set an objects origin to the center of its bounding box
    :param o:
    :return:
    """
    me = o.data
    bounds = _calculate_bounds(me.vertices)
    if bounds is None:
        return

    min_x, max_x, min_y, max_y, min_z, max_z = bounds
    set_origin(o, Vector(((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2)))


def set_origin_min_bounds(o):
    """
    set an objects origin to the bottom_left corner of its bounding box
    :param o:
    :return:
    """
    me = o.data
    bounds = _calculate_bounds(me.vertices)
    if bounds is None:
        return

    min_x, _, min_y, _, min_z, max_z = bounds
    set_origin(o, Vector((min_x, min_y, (min_z + max_z) / 2)))


def create_mesh(context, vertices, faces, name):
    verts = [Vector(i) for i in vertices]

    # Blender can handle n-gons directly, no need for createPolys
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    return object_data_add(context, mesh, operator=None)


def ensure_material(name, base_color):
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    if material.node_tree:
        bsdf = material.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = base_color
    return material


def assign_material(obj, material):
    if obj.data.materials:
        obj.data.materials.clear()
    obj.data.materials.append(material)


def apply_transform(ob, use_location=False, use_rotation=False, use_scale=False):
    """
    https://blender.stackexchange.com/questions/159538/how-to-apply-all-transformations-to-an-object-at-low-level
    :param ob:
    :param use_location:
    :param use_rotation:
    :param use_scale:
    :return:
    """
    mb = ob.matrix_basis
    I = Matrix()
    loc, rot, scale = mb.decompose()

    # rotation
    T = Matrix.Translation(loc)
    # R = rot.to_matrix().to_4x4()
    R = mb.to_3x3().normalized().to_4x4()
    S = Matrix.Diagonal(scale).to_4x4()

    transform = [I, I, I]
    basis = [T, R, S]

    def swap(i):
        transform[i], basis[i] = basis[i], transform[i]

    if use_location:
        swap(0)
    if use_rotation:
        swap(1)
    if use_scale:
        swap(2)

    M = transform[0] @ transform[1] @ transform[2]
    if hasattr(ob.data, "transform"):
        ob.data.transform(M)
    for c in ob.children:
        c.matrix_local = M @ c.matrix_local

    ob.matrix_basis = basis[0] @ basis[1] @ basis[2]


def join(objects):
    """Join a list of objects into one and return the result."""
    if not objects:
        return None

    view_layer = bpy.context.view_layer

    # Deselect everything first
    for ob in view_layer.objects:
        ob.select_set(False)

    # Select the objects we want to join
    for ob in objects:
        ob.select_set(True)

    # Set the active object (the one that will remain after the join)
    view_layer.objects.active = objects[0]

    # Run the join operator in the current context
    bpy.ops.object.join()

    return objects[0]


def validate_font_path(filepath):
    # set font to emtpy if it's not a ttf file
    if filepath and not os.path.isfile(filepath):
        return ''

    if filepath and os.path.splitext(filepath)[1].lower() not in ('.ttf', '.otf'):
        return ''
    return filepath


def validate_svg_path(filepath):
    if filepath and not os.path.isfile(filepath):
        return ''

    if filepath and os.path.splitext(filepath)[1].lower() != '.svg':
        return ''

    return filepath


SETTINGS_ATTRS = [
    "size",
    "dice_finish",
    "bumper_scale",
    "font_path",
    "number_scale",
    "number_depth",
    "one_offset",
    "add_numbers",
    "number_indicator_type",
    "period_indicator_scale",
    "period_indicator_space",
    "bar_indicator_height",
    "bar_indicator_width",
    "bar_indicator_space",
    "center_bar",
    "number_v_offset",
    "number_center_offset",
    "base_height",
    "point_height",
    "top_point_height",
    "bottom_point_height",
    "height",
    "custom_image_path",
    "custom_image_face",
    "custom_image_scale",
]


def collect_settings_from_op(op, settings_template):
    return {attr: getattr(op, attr, getattr(settings_template, attr)) for attr in SETTINGS_ATTRS}


def apply_settings(settings_obj, values):
    for key, value in values.items():
        setattr(settings_obj, key, value)


def snapshot_settings(settings_obj):
    return {attr: getattr(settings_obj, attr) for attr in SETTINGS_ATTRS}


def resolve_settings_owner(obj):
    if obj is None or not hasattr(obj, "dice_gen_settings"):
        return None

    if obj.get("dice_gen_type") is not None:
        return obj

    numbers_name = obj.get("dice_numbers_name")
    if numbers_name and numbers_name in bpy.data.objects:
        numbers_obj = bpy.data.objects[numbers_name]
        if numbers_obj.get("dice_gen_type") is not None:
            return numbers_obj

    return None


def get_font(filepath):
    if filepath:
        try:
            bpy.data.fonts.load(filepath=filepath, check_existing=True)
            return next(filter(lambda x: x.filepath == filepath, bpy.data.fonts))
        except (RuntimeError, OSError):
            pass

    bpy.data.fonts.load(filepath='<builtin>', check_existing=True)
    return bpy.data.fonts[0]


def apply_boolean_modifier(body_object, numbers_object):
    """
    Add a BOOLEAN modifier to body_object that targets
    :param context:
    :param body_object:
    :param numbers_object
    :return:
    """
    numbers_boolean = body_object.modifiers.new(type='BOOLEAN', name='boolean')
    numbers_boolean.object = bpy.data.objects[numbers_object.name]
    numbers_boolean.show_viewport = False

    # remember the numbers object for regeneration
    body_object["dice_numbers_name"] = numbers_object.name


@contextmanager
def ensure_object_mode(active_obj):
    """Temporarily switch to OBJECT mode for mesh edits and restore the prior mode."""
    view_layer = bpy.context.view_layer
    previous_active = view_layer.objects.active
    previous_mode = active_obj.mode if active_obj else None

    try:
        if active_obj and view_layer.objects.active != active_obj:
            view_layer.objects.active = active_obj

        if active_obj and active_obj.mode != 'OBJECT':
            try:
                bpy.ops.object.mode_set(mode='OBJECT')
            except RuntimeError:
                pass

        yield
    finally:
        if active_obj and previous_mode and previous_mode != 'OBJECT':
            try:
                bpy.ops.object.mode_set(mode=previous_mode)
            except RuntimeError:
                pass

        if previous_active and previous_active != view_layer.objects.active:
            view_layer.objects.active = previous_active


def apply_bumpers_to_mesh(mesh_data, bumper_scale):
    inset_amount = 0.3 * bumper_scale
    extrude_amount = 0.5 * bumper_scale

    if inset_amount <= 0 and extrude_amount <= 0:
        return

    bm = bmesh.new()
    bm.from_mesh(mesh_data)

    if not bm.faces:
        bm.free()
        return

    inset_result = bmesh.ops.inset_individual(
        bm,
        faces=list(bm.faces),
        thickness=inset_amount,
        depth=0.0,
        use_even_offset=True,
    )

    inset_faces = set(inset_result.get("faces", []))
    rim_faces = [face for face in bm.faces if face not in inset_faces]

    if extrude_amount > 0:
        for rim_face in rim_faces:
            normal = rim_face.normal.copy()
            if normal.length == 0:
                continue

            extrude_result = bmesh.ops.extrude_faces_individual(bm, faces=[rim_face])
            extruded_geom = extrude_result.get("geom", [])
            extruded_verts = [
                elem for elem in extruded_geom if isinstance(elem, bmesh.types.BMVert)
            ]

            if extruded_verts:
                bmesh.ops.translate(
                    bm,
                    verts=extruded_verts,
                    vec=normal.normalized() * extrude_amount,
                )

    bm.normal_update()
    bm.to_mesh(mesh_data)
    bm.free()


def configure_dice_finish_modifier(body_object, dice_finish, bumper_scale=1):
    if body_object is None or body_object.type != 'MESH':
        return

    with ensure_object_mode(body_object):
        modifier_name = "dice_bevel"
        bevel_modifier = body_object.modifiers.get(modifier_name)
        bumper_base_key = "dice_base_mesh_name"

        if dice_finish != "bumpers":
            base_mesh_name = body_object.get(bumper_base_key)
            if base_mesh_name and base_mesh_name in bpy.data.meshes:
                base_mesh = bpy.data.meshes[base_mesh_name]
                if body_object.data != base_mesh:
                    previous_mesh = body_object.data
                    body_object.data = base_mesh.copy()
                    if previous_mesh.users == 0 and previous_mesh != base_mesh:
                        bpy.data.meshes.remove(previous_mesh)

                base_mesh.use_fake_user = False
                if base_mesh.users == 0:
                    bpy.data.meshes.remove(base_mesh)

                if bumper_base_key in body_object:
                    del body_object[bumper_base_key]

        if dice_finish == "bumpers":
            if bevel_modifier:
                body_object.modifiers.remove(bevel_modifier)

            base_mesh = None
            base_mesh_name = body_object.get(bumper_base_key)

            if base_mesh_name and base_mesh_name in bpy.data.meshes:
                base_mesh = bpy.data.meshes[base_mesh_name]
            else:
                base_mesh = body_object.data.copy()
                base_mesh.use_fake_user = True
                body_object[bumper_base_key] = base_mesh.name

            working_mesh = base_mesh.copy()
            apply_bumpers_to_mesh(working_mesh, bumper_scale)

            previous_mesh = body_object.data
            body_object.data = working_mesh

            if previous_mesh not in (base_mesh, working_mesh) and previous_mesh.users == 0:
                bpy.data.meshes.remove(previous_mesh)

            return

        if dice_finish == "sharp":
            if bevel_modifier:
                body_object.modifiers.remove(bevel_modifier)
            return

        if bevel_modifier is None:
            bevel_modifier = body_object.modifiers.new(type='BEVEL', name=modifier_name)

        bevel_modifier.limit_method = 'NONE'
        bevel_modifier.use_clamp_overlap = False
        bevel_modifier.width = 0.3
        bevel_modifier.segments = 1 if dice_finish == "chamfer" else 5

        if hasattr(bevel_modifier, "affect"):
            bevel_modifier.affect = 'EDGES'


def create_svg_mesh(context, filepath, scale, depth, name):
    existing_objects = set(bpy.data.objects)
    existing_collections = set(bpy.data.collections)
    new_collections = []

    try:
        bpy.ops.import_curve.svg(filepath=filepath)
    except (RuntimeError, OSError):
        return None

    new_collections = [col for col in bpy.data.collections if col not in existing_collections]
    imported_objects = [ob for ob in bpy.data.objects if ob not in existing_objects]
    imported_object_names = [ob.name for ob in imported_objects]
    curve_object_names = [ob.name for ob in imported_objects if ob.type == 'CURVE']
    curve_objects = [bpy.data.objects[name] for name in curve_object_names if name in bpy.data.objects]

    def cleanup_new_collections():
        for collection in new_collections:
            if collection.objects or collection.children:
                continue

            for parent in bpy.data.collections:
                if parent.children.get(collection.name):
                    parent.children.unlink(collection)

            for scene in bpy.data.scenes:
                if scene.collection.children.get(collection.name):
                    scene.collection.children.unlink(collection)

            try:
                bpy.data.collections.remove(collection)
            except RuntimeError:
                # If the collection still has users for any reason, skip removal
                pass

    if not curve_objects:
        for obj_name in imported_object_names:
            if obj_name in bpy.data.objects:
                bpy.data.objects.remove(bpy.data.objects[obj_name], do_unlink=True)
        cleanup_new_collections()
        return None

    mesh_objects = []
    for curve_obj in curve_objects:
        curve_obj.data.materials.clear()
        if hasattr(curve_obj.data, "color_attributes"):
            for color_attr in list(curve_obj.data.color_attributes):
                curve_obj.data.color_attributes.remove(color_attr)

        curve_obj.data.extrude = depth

        mesh = curve_obj.to_mesh().copy()
        mesh.materials.clear()
        if hasattr(mesh, "color_attributes"):
            for color_attr in list(mesh.color_attributes):
                mesh.color_attributes.remove(color_attr)
        new_obj = object_data_add(context, mesh, operator=None)
        mesh_objects.append(new_obj)

    for curve_name in curve_object_names:
        if curve_name in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[curve_name], do_unlink=True)

    if not mesh_objects:
        cleanup_new_collections()
        return None

    for obj_name in imported_object_names:
        if obj_name in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[obj_name], do_unlink=True)

    cleanup_new_collections()

    svg_mesh = join(mesh_objects)
    svg_mesh.name = name

    current_dimension = max(svg_mesh.dimensions.x, svg_mesh.dimensions.y)
    current_dimension = current_dimension if current_dimension else 1
    uniform_scale = scale / current_dimension
    svg_mesh.scale = (uniform_scale, uniform_scale, 1)

    apply_transform(svg_mesh, use_scale=True)
    return svg_mesh


def create_text_mesh(context, text, font_path, font_size, name, extrude=0):
    # load the font
    font = get_font(font_path)

    # create the text curve
    font_curve = bpy.data.curves.new(type='FONT', name=name)
    font_curve.body = text
    font_curve.font = font
    font_curve.size = font_size
    font_curve.extrude = extrude
    font_curve.offset = 0

    # create object from curve
    curve_obj = bpy.data.objects.new('temp_curve_obj', font_curve)

    # convert curve to mesh
    mesh = curve_obj.to_mesh().copy()
    curve_obj.to_mesh_clear()
    bpy.data.objects.remove(curve_obj)
    bpy.data.curves.remove(font_curve)
    return object_data_add(context, mesh, operator=None)


def create_numbers(context, numbers, locations, rotations, font_path, font_size, number_depth, number_indicator_type,
                   period_indicator_scale, period_indicator_space, bar_indicator_height, bar_indicator_width,
                   bar_indicator_space, center_bar, one_offset, custom_image_face=0, custom_image_path='',
                   custom_image_scale=1):
    number_objs = []
    # create the number meshes
    for i in range(len(locations)):
        number_object = create_number(context, numbers[i], font_path, font_size, number_depth, locations[i],
                                      rotations[i], number_indicator_type, period_indicator_scale,
                                      period_indicator_space, bar_indicator_height, bar_indicator_width,
                                      bar_indicator_space, center_bar, one_offset,
                                      custom_image_face=custom_image_face, custom_image_path=custom_image_path,
                                      custom_image_scale=custom_image_scale, index=i)
        number_objs.append(number_object)

    # join the numbers into a single object
    if len(number_objs):
        numbers = join(number_objs)
        apply_transform(numbers, use_rotation=True, use_location=True)
        numbers_material = ensure_material("Dice Numbers", (0, 0, 0, 1))
        assign_material(numbers, numbers_material)
        return numbers

    return None


def create_number(context, number, font_path, font_size, number_depth, location, rotation, number_indicator_type,
                  period_indicator_scale, period_indicator_space, bar_indicator_height, bar_indicator_width,
                  bar_indicator_space, center_bar, one_offset, custom_image_face=0, custom_image_path='',
                  custom_image_scale=1, index=0):
    """
    Create a number mesh that will be used in a boolean modifier
    """
    use_custom_image = custom_image_path and (custom_image_face == index + 1)

    mesh_object = None

    if use_custom_image:
        mesh_object = create_svg_mesh(context, custom_image_path, font_size * custom_image_scale, number_depth,
                                      f'custom_image_{index + 1}')

    if mesh_object is None:
        # add number
        mesh_object = create_text_mesh(context, number, font_path, font_size, f'number_{number}', number_depth)

    # set origin to bounding box center
    set_origin_center_bounds(mesh_object)

    if not use_custom_image:
        if number == '1':
            if one_offset > 0:
                number_width = mesh_object.dimensions.x
                new_origin = Vector((mesh_object.location.x + number_width * one_offset, mesh_object.location.y,
                                     mesh_object.location.z))
                set_origin(mesh_object, new_origin)
                pass
        elif number in ('6', '9'):
            if number_indicator_type == NUMBER_IND_PERIOD:
                p_obj = create_text_mesh(context, '.', font_path, font_size * period_indicator_scale, f'period_{number}',
                                         number_depth)

                # move origin of period to the bottom left corner of the mesh
                set_origin_min_bounds(p_obj)

                space = (1 / 20) * font_size * period_indicator_space

                # move period to the bottom right of the number
                p_obj.location = Vector((mesh_object.location.x + (mesh_object.dimensions.x / 2) + space,
                                         mesh_object.location.y - (mesh_object.dimensions.y / 2), 0))

                # join the period to the number
                mesh_object = join([mesh_object, p_obj])
            elif number_indicator_type == NUMBER_IND_BAR:
                # create a simple rectangle
                bar_width = mesh_object.dimensions.x * bar_indicator_width
                bar_height = (1 / 15) * font_size * bar_indicator_height
                bar_space = (1 / 20) * font_size * bar_indicator_space
                bar_obj = create_mesh(context,
                                      [(-bar_width / 2, -bar_space, number_depth),
                                       (bar_width / 2, -bar_space, number_depth),
                                       (-bar_width / 2, -bar_space - bar_height, number_depth),
                                       (bar_width / 2, -bar_space - bar_height, number_depth),
                                       (-bar_width / 2, -bar_space, -number_depth),
                                       (bar_width / 2, -bar_space, -number_depth),
                                       (-bar_width / 2, -bar_space - bar_height, -number_depth),
                                       (bar_width / 2, -bar_space - bar_height, -number_depth)],
                                      [[0, 1, 3, 2], [2, 3, 7, 6], [3, 1, 5, 7], [1, 0, 4, 5], [0, 2, 6, 4], [4, 6, 7, 5]],
                                      'bar_indicator')

                # move bar below the number
                bar_obj.location = Vector(
                    (mesh_object.location.x, mesh_object.location.y - (mesh_object.dimensions.y / 2), 0))

                # join the bar to the number
                mesh_object = join([mesh_object, bar_obj])

                # recenter the mesh
                if center_bar:
                    mesh_object.location = Vector((0, 0, 0))
                    set_origin_center_bounds(mesh_object)

    mesh_object.location.x = location[0]
    mesh_object.location.y = location[1]
    mesh_object.location.z = location[2]

    mesh_object.rotation_euler.x = rotation[0]
    mesh_object.rotation_euler.y = rotation[1]
    mesh_object.rotation_euler.z = rotation[2]

    for f in mesh_object.data.polygons:
        f.use_smooth = False

    return mesh_object


def execute_generator(op, context, mesh_cls, name, **kwargs):
    # set font to emtpy if it's not a ttf file
    op.font_path = validate_font_path(op.font_path)
    op.custom_image_path = validate_svg_path(op.custom_image_path)

    # create the cube mesh
    die = mesh_cls("dice_body", op.size, **kwargs)
    die_obj = die.create(context)
    configure_dice_finish_modifier(die_obj, op.dice_finish, getattr(op, "bumper_scale", 1))
    body_material = ensure_material("Dice Body", (0.95, 0.95, 0.9, 1))
    assign_material(die_obj, body_material)

    settings_template = die_obj.dice_gen_settings
    settings_values = collect_settings_from_op(op, settings_template)

    numbers_object = None
    # create number curves
    if op.add_numbers:
        if op.number_indicator_type == NUMBER_IND_NONE:
            numbers_object = die.create_numbers(
                context, op.size, op.number_scale, op.number_depth, op.font_path, op.one_offset,
                custom_image_face=op.custom_image_face, custom_image_path=op.custom_image_path,
                custom_image_scale=op.custom_image_scale
            )
        else:
            numbers_object = die.create_numbers(
                context, op.size, op.number_scale, op.number_depth, op.font_path, op.one_offset,
                op.number_indicator_type, op.period_indicator_scale, op.period_indicator_space,
                op.bar_indicator_height, op.bar_indicator_width, op.bar_indicator_space, op.center_bar,
                custom_image_face=op.custom_image_face, custom_image_path=op.custom_image_path,
                custom_image_scale=op.custom_image_scale
            )

    target_object = numbers_object or die_obj
    target_object["dice_gen_type"] = mesh_cls.__name__
    if numbers_object is not None:
        numbers_object["dice_body_name"] = die_obj.name

    apply_settings(target_object.dice_gen_settings, settings_values)

    return {'FINISHED'}


# Common properties
def Face2FaceProperty(default: float):
    return FloatProperty(
        name='Face2face Length',
        description='Face-to-face size of the die',
        min=1,
        soft_min=1,
        max=100,
        soft_max=100,
        default=default,
        unit='LENGTH'
    )


def DiceFinishProperty():
    return EnumProperty(
        name='Dice Type',
        items=(
            ('sharp', 'Sharp', 'Keep edges sharp'),
            ('chamfer', 'Chamfer', 'Add a light bevel'),
            ('fillet', 'Fillet', 'Round edges with additional bevel segments'),
            ('bumpers', 'Bumpers', 'Inset faces and raise the face borders'),
        ),
        default='sharp',
        description='Edge treatment for the dice body'
    )


def BumperScaleProperty():
    return FloatProperty(
        name='Bumper Size',
        description='Scale the inset and extrusion used to create bumper edges',
        min=0,
        soft_min=0,
        max=5,
        soft_max=5,
        default=1,
    )


AddNumbersProperty = BoolProperty(
    name='Generate Numbers',
    default=True
)

NumberScaleProperty = FloatProperty(
    name='Number Scale',
    description='Size of the numbers on the die',
    min=0.1,
    soft_min=0.1,
    max=2,
    soft_max=2,
    default=1
)

NumberDepthProperty = FloatProperty(
    name='Number Depth',
    description='Depth of the numbers on the die',
    min=0.1,
    soft_min=0.1,
    max=2,
    soft_max=2,
    default=0.75,
    unit='LENGTH'
)

FontPathProperty = StringProperty(
    name='Font',
    description='Number font',
    maxlen=1024,
    subtype='FILE_PATH'
)

CustomImagePathProperty = StringProperty(
    name='Custom Image (SVG)',
    description='SVG file to engrave on a selected face',
    maxlen=1024,
    subtype='FILE_PATH'
)

CustomImageFaceProperty = IntProperty(
    name='Custom Image Face',
    description='1-based face index to replace with the custom image (0 disables the feature)',
    min=0,
    soft_min=0,
    default=0
)

CustomImageScaleProperty = FloatProperty(
    name='Custom Image Scale',
    description='Scale multiplier for the custom image relative to the number size',
    min=0.01,
    soft_min=0.01,
    max=10,
    soft_max=10,
    default=1
)

OneOffsetProperty = FloatProperty(
    name='Number 1 Offset',
    description='Offset the number 1 horizontally for an alternative centering',
    min=0,
    soft_min=0,
    max=1,
    soft_max=1,
    default=0
)


# Indicator properties
def NumberIndicatorTypeProperty(default: str = NUMBER_IND_PERIOD):
    return EnumProperty(
        name='Orientation Indicator',
        items=((NUMBER_IND_NONE, 'None', ','),
               (NUMBER_IND_BAR, 'Bar', ''),
               (NUMBER_IND_PERIOD, 'Period', '')),
        default=default,
        description='Orientation indicator for numbers 6 and 9'
    )


PeriodIndicatorScaleProperty = FloatProperty(
    name='Period Scale',
    description='Scale of the period orientation indicator',
    min=0.1,
    soft_min=0.1,
    max=2,
    soft_max=2,
    default=1
)

PeriodIndicatorSpaceProperty = FloatProperty(
    name='Period Space',
    description='Space between the period orientation indicator and the number',
    min=0,
    soft_min=0,
    max=3,
    soft_max=3,
    default=1
)

BarIndicatorHeightProperty = FloatProperty(
    name='Bar Height',
    description='Height scale of the bar orientation indicator',
    min=0.1,
    soft_min=0.1,
    max=3,
    soft_max=3,
    default=1
)

BarIndicatorWidthProperty = FloatProperty(
    name='Bar Width',
    description='Width scale of the bar orientation indicator',
    min=0.1,
    soft_min=0.1,
    max=2,
    soft_max=2,
    default=1
)

BarIndicatorSpaceProperty = FloatProperty(
    name='Bar Space',
    description='Space between the bar orientation indicator and the number',
    min=0,
    soft_min=0,
    max=3,
    soft_max=3,
    default=1
)

CenterBarProperty = BoolProperty(
    name='Center Align Bar',
    description='If true, the bar indicator is included in the vertical alignment of the number',
    default=True
)



def NumberVOffsetProperty(default: float): return FloatProperty(
    name='Number V Offset',
    description='Vertical offset of the number positioning',
    min=0.0,
    soft_min=0.0,
    max=1,
    soft_max=1,
    default=default
)


class DiceGenSettings(bpy.types.PropertyGroup):
    size: FloatProperty(
        name="Face2Face Length",
        description="Face-to-face size of the die",
        min=1,
        soft_min=1,
        max=100,
        soft_max=100,
        default=20,
        unit='LENGTH',
    )

    dice_finish: DiceFinishProperty()

    bumper_scale: BumperScaleProperty()

    font_path: FontPathProperty

    custom_image_path: CustomImagePathProperty

    custom_image_face: CustomImageFaceProperty

    custom_image_scale: CustomImageScaleProperty

    number_scale: NumberScaleProperty

    number_depth: NumberDepthProperty

    one_offset: OneOffsetProperty

    add_numbers: AddNumbersProperty

    number_indicator_type: NumberIndicatorTypeProperty()

    period_indicator_scale: PeriodIndicatorScaleProperty

    period_indicator_space: PeriodIndicatorSpaceProperty

    bar_indicator_height: BarIndicatorHeightProperty

    bar_indicator_width: BarIndicatorWidthProperty

    bar_indicator_space: BarIndicatorSpaceProperty

    center_bar: CenterBarProperty

    number_v_offset: NumberVOffsetProperty(1 / 3)

    number_center_offset: FloatProperty(
        name='Number Center Offset',
        description='Distance of numbers from the center of a face',
        min=0.0,
        soft_min=0.0,
        max=1,
        soft_max=1,
        default=0.5
    )

    base_height: FloatProperty(
        name='Base Height',
        description='Base height of the die (height of a face)',
        min=1,
        soft_min=1,
        max=100,
        soft_max=100,
        default=14,
        unit='LENGTH',
    )

    point_height: FloatProperty(
        name='Point Height',
        description='Point height of the die',
        min=1,
        soft_min=1,
        max=100,
        soft_max=100,
        default=7,
        unit='LENGTH',
    )

    top_point_height: FloatProperty(
        name='Top Point Height',
        description='Top point height of the die',
        min=0.25,
        soft_min=0.25,
        max=2,
        soft_max=2,
        default=0.75
    )

    bottom_point_height: FloatProperty(
        name='Bottom Point Height',
        description='Bottom point height of the die',
        min=0.25,
        soft_min=0.25,
        max=2.5,
        soft_max=2.5,
        default=1.75
    )

    height: FloatProperty(
        name='Dice Height',
        description='Height of the die',
        min=0.45,
        soft_min=0.45,
        max=2,
        soft_max=2,
        default=2 / 3
    )


class DiceGeneratorBase:
    dice_finish: DiceFinishProperty()
    bumper_scale: BumperScaleProperty()

    def draw(self, context):
        layout = self.layout

        layout.prop(self, "dice_finish")

        seen_props = {"dice_finish"}
        for cls in reversed(type(self).mro()):
            annotations = getattr(cls, "__annotations__", {})
            for prop_name in annotations:
                if prop_name in seen_props:
                    continue

                if prop_name == "bumper_scale" and self.dice_finish != "bumpers":
                    continue

                if hasattr(self, prop_name):
                    layout.prop(self, prop_name)
                    seen_props.add(prop_name)


class D4Generator(DiceGeneratorBase, bpy.types.Operator):
    """Generate a D4"""
    bl_idname = 'mesh.d4_add'
    bl_label = 'D4 Tetrahedron'
    bl_description = 'Generate a tetrahedron dice'
    bl_options = {'REGISTER', 'UNDO'}

    number_indicator_type = NUMBER_IND_NONE

    size: FloatProperty(
        name='Face2Point Length',
        description='Face-to-point size of the die',
        min=1,
        soft_min=1,
        max=100,
        soft_max=100,
        default=20,
        unit='LENGTH'
    )

    add_numbers: AddNumbersProperty

    number_scale: NumberScaleProperty

    number_depth: NumberDepthProperty

    font_path: FontPathProperty

    custom_image_path: CustomImagePathProperty

    custom_image_face: CustomImageFaceProperty

    custom_image_scale: CustomImageScaleProperty

    one_offset: OneOffsetProperty

    number_center_offset: FloatProperty(
        name='Number Center Offset',
        description='Distance of numbers from the center of a face',
        min=0.0,
        soft_min=0.0,
        max=1,
        soft_max=1,
        default=0.5
    )

    def execute(self, context):
        return execute_generator(self, context, Tetrahedron, 'd4', number_center_offset=self.number_center_offset)


class D4CrystalGenerator(DiceGeneratorBase, bpy.types.Operator):
    """Generate a D4 crystal"""
    bl_idname = 'mesh.d4_crystal_add'
    bl_label = 'D4 Crystal'
    bl_description = 'Generate a D4 crystal dice'
    bl_options = {'REGISTER', 'UNDO'}

    number_indicator_type = NUMBER_IND_NONE

    size: Face2FaceProperty(12)

    base_height: FloatProperty(
        name='Base Height',
        description='Base height of the die (height of a face)',
        min=1,
        soft_min=1,
        max=100,
        soft_max=100,
        default=14,
        unit='LENGTH'
    )

    point_height: FloatProperty(
        name='Point Height',
        description='Point height of the die',
        min=1,
        soft_min=1,
        max=100,
        soft_max=100,
        default=7,
        unit='LENGTH'
    )

    add_numbers: AddNumbersProperty
    number_scale: NumberScaleProperty
    number_depth: NumberDepthProperty
    font_path: FontPathProperty
    one_offset: OneOffsetProperty
    custom_image_path: CustomImagePathProperty
    custom_image_face: CustomImageFaceProperty
    custom_image_scale: CustomImageScaleProperty

    def execute(self, context):
        return execute_generator(self, context, D4Crystal, 'd4Crystal', base_height=self.base_height,
                                 point_height=self.point_height)


class D4ShardGenerator(DiceGeneratorBase, bpy.types.Operator):
    """Generate a D4 shard"""
    bl_idname = 'mesh.d4_shard_add'
    bl_label = 'D4 Shard'
    bl_description = 'Generate a D4 crystal dice'
    bl_options = {'REGISTER', 'UNDO'}

    number_indicator_type = NUMBER_IND_NONE

    size: FloatProperty(
        name='Edge2edge length',
        description='Distance between 2 opposite horizontal edges',
        min=1,
        soft_min=1,
        max=100,
        soft_max=100,
        default=12,
        unit='LENGTH'
    )

    top_point_height: FloatProperty(
        name='Top Point Height',
        description='Top point height of the die',
        min=0.25,
        soft_min=0.25,
        max=2,
        soft_max=2,
        default=0.75
    )

    bottom_point_height: FloatProperty(
        name='Bottom Point Height',
        description='Bottom point height of the die',
        min=0.25,
        soft_min=0.25,
        max=2.5,
        soft_max=2.5,
        default=1.75
    )

    add_numbers: AddNumbersProperty
    number_scale: NumberScaleProperty
    number_depth: NumberDepthProperty
    font_path: FontPathProperty
    one_offset: OneOffsetProperty
    number_v_offset: NumberVOffsetProperty(0.7)
    custom_image_path: CustomImagePathProperty
    custom_image_face: CustomImageFaceProperty
    custom_image_scale: CustomImageScaleProperty

    def execute(self, context):
        return execute_generator(self, context, D4Shard, 'd4Shard', top_point_height=self.top_point_height,
                                 bottom_point_height=self.bottom_point_height, number_v_offset=self.number_v_offset)


class D6Generator(DiceGeneratorBase, bpy.types.Operator):
    """Generate a D6"""
    bl_idname = 'mesh.d6_add'
    bl_label = 'D6 Cube'
    bl_description = 'Generate a cube dice'
    bl_options = {'REGISTER', 'UNDO'}

    size: Face2FaceProperty(16)
    add_numbers: AddNumbersProperty
    number_scale: NumberScaleProperty
    number_depth: NumberDepthProperty
    font_path: FontPathProperty
    one_offset: OneOffsetProperty
    number_indicator_type: NumberIndicatorTypeProperty(NUMBER_IND_NONE)
    period_indicator_scale: PeriodIndicatorScaleProperty
    period_indicator_space: PeriodIndicatorSpaceProperty
    bar_indicator_height: BarIndicatorHeightProperty
    bar_indicator_width: BarIndicatorWidthProperty
    bar_indicator_space: BarIndicatorSpaceProperty
    center_bar: CenterBarProperty
    custom_image_path: CustomImagePathProperty
    custom_image_face: CustomImageFaceProperty
    custom_image_scale: CustomImageScaleProperty

    def execute(self, context):
        return execute_generator(self, context, Cube, 'd6')


class D8Generator(DiceGeneratorBase, bpy.types.Operator):
    """Generate a D8"""
    bl_idname = 'mesh.d8_add'
    bl_label = 'D8 Octahedron'
    bl_description = 'Generate a octahedron dice'
    bl_options = {'REGISTER', 'UNDO'}

    size: Face2FaceProperty(15)
    add_numbers: AddNumbersProperty
    number_scale: NumberScaleProperty
    number_depth: NumberDepthProperty
    font_path: FontPathProperty
    one_offset: OneOffsetProperty
    number_indicator_type: NumberIndicatorTypeProperty(NUMBER_IND_NONE)
    period_indicator_scale: PeriodIndicatorScaleProperty
    period_indicator_space: PeriodIndicatorSpaceProperty
    bar_indicator_height: BarIndicatorHeightProperty
    bar_indicator_width: BarIndicatorWidthProperty
    bar_indicator_space: BarIndicatorSpaceProperty
    center_bar: CenterBarProperty
    custom_image_path: CustomImagePathProperty
    custom_image_face: CustomImageFaceProperty
    custom_image_scale: CustomImageScaleProperty

    def execute(self, context):
        return execute_generator(self, context, Octahedron, 'd8')


class D12Generator(DiceGeneratorBase, bpy.types.Operator):
    """Generate a D12"""
    bl_idname = 'mesh.d12_add'
    bl_label = 'D12 Dodecahedron'
    bl_description = 'Generate a dodecahedron dice'
    bl_options = {'REGISTER', 'UNDO'}

    size: Face2FaceProperty(18)
    add_numbers: AddNumbersProperty
    number_scale: NumberScaleProperty
    number_depth: NumberDepthProperty
    font_path: FontPathProperty
    one_offset: OneOffsetProperty
    number_indicator_type: NumberIndicatorTypeProperty()
    period_indicator_scale: PeriodIndicatorScaleProperty
    period_indicator_space: PeriodIndicatorSpaceProperty
    bar_indicator_height: BarIndicatorHeightProperty
    bar_indicator_width: BarIndicatorWidthProperty
    bar_indicator_space: BarIndicatorSpaceProperty
    center_bar: CenterBarProperty
    custom_image_path: CustomImagePathProperty
    custom_image_face: CustomImageFaceProperty
    custom_image_scale: CustomImageScaleProperty

    def execute(self, context):
        return execute_generator(self, context, Dodecahedron, 'd12')


class D20Generator(DiceGeneratorBase, bpy.types.Operator):
    """Generate a D20"""
    bl_idname = 'mesh.d20_add'
    bl_label = 'D20 Icosahedron'
    bl_description = 'Generate an icosahedron dice'
    bl_options = {'REGISTER', 'UNDO'}

    size: Face2FaceProperty(20)
    add_numbers: AddNumbersProperty
    number_scale: NumberScaleProperty
    number_depth: NumberDepthProperty
    font_path: FontPathProperty
    one_offset: OneOffsetProperty
    number_indicator_type: NumberIndicatorTypeProperty()
    period_indicator_scale: PeriodIndicatorScaleProperty
    period_indicator_space: PeriodIndicatorSpaceProperty
    bar_indicator_height: BarIndicatorHeightProperty
    bar_indicator_width: BarIndicatorWidthProperty
    bar_indicator_space: BarIndicatorSpaceProperty
    center_bar: CenterBarProperty
    custom_image_path: CustomImagePathProperty
    custom_image_face: CustomImageFaceProperty
    custom_image_scale: CustomImageScaleProperty

    def execute(self, context):
        return execute_generator(self, context, Icosahedron, 'd20')


class D10Generator(DiceGeneratorBase, bpy.types.Operator):
    """Generate a D10"""
    bl_idname = 'mesh.d10_add'
    bl_label = 'D10 Trapezohedron'
    bl_description = 'Generate an d10 trapezohedron dice'
    bl_options = {'REGISTER', 'UNDO'}

    size: Face2FaceProperty(17)

    height: FloatProperty(
        name='Dice Height',
        description='Height of the die',
        min=0.45,
        soft_min=0.45,
        max=2,
        soft_max=2,
        default=2 / 3
    )

    add_numbers: AddNumbersProperty
    number_scale: NumberScaleProperty
    number_depth: NumberDepthProperty
    font_path: FontPathProperty
    one_offset: OneOffsetProperty
    number_v_offset: NumberVOffsetProperty(1 / 3)
    number_indicator_type: NumberIndicatorTypeProperty()
    period_indicator_scale: PeriodIndicatorScaleProperty
    period_indicator_space: PeriodIndicatorSpaceProperty
    bar_indicator_height: BarIndicatorHeightProperty
    bar_indicator_width: BarIndicatorWidthProperty
    bar_indicator_space: BarIndicatorSpaceProperty
    center_bar: CenterBarProperty
    custom_image_path: CustomImagePathProperty
    custom_image_face: CustomImageFaceProperty
    custom_image_scale: CustomImageScaleProperty

    def execute(self, context):
        return execute_generator(self, context, D10Mesh, 'd10', height=self.height,
                                 number_v_offset=self.number_v_offset)


class D100Generator(DiceGeneratorBase, bpy.types.Operator):
    """Generate a D100"""
    bl_idname = 'mesh.d100_add'
    bl_label = 'D100 Trapezohedron'
    bl_description = 'Generate an d100 trapezohedron dice'
    bl_options = {'REGISTER', 'UNDO'}

    number_indicator_type = NUMBER_IND_NONE
    one_offset = 0

    size: Face2FaceProperty(17)

    height: FloatProperty(
        name='Dice Height',
        description='Height of the die',
        min=0.45,
        soft_min=0.45,
        max=2,
        soft_max=2,
        default=2 / 3
    )

    add_numbers: AddNumbersProperty
    number_scale: NumberScaleProperty
    number_depth: NumberDepthProperty
    font_path: FontPathProperty
    number_v_offset: NumberVOffsetProperty(1 / 3)
    custom_image_path: CustomImagePathProperty
    custom_image_face: CustomImageFaceProperty
    custom_image_scale: CustomImageScaleProperty

    def execute(self, context):
        return execute_generator(self, context, D100Mesh, 'd100', height=self.height,
                                 number_v_offset=self.number_v_offset)


class OBJECT_OT_dice_gen_update(bpy.types.Operator):
    bl_idname = "object.dice_gen_update"
    bl_label = "Update Dice Numbers"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        ob = context.object
        settings_owner = resolve_settings_owner(ob)

        if settings_owner is None:
            self.report({'ERROR'}, "Object has no dice settings")
            return {'CANCELLED'}

        settings_values = snapshot_settings(settings_owner.dice_gen_settings)
        die_type = settings_owner.get("dice_gen_type")

        numbers_obj = None
        if settings_owner.get("dice_body_name"):
            numbers_obj = settings_owner
        elif ob.get("dice_numbers_name"):
            numbers_obj = bpy.data.objects.get(ob.get("dice_numbers_name"))

        body_obj = None
        if settings_owner.get("dice_body_name"):
            body_obj = bpy.data.objects.get(settings_owner.get("dice_body_name"))
        else:
            body_obj = ob

        if body_obj is None or die_type is None:
            self.report({'ERROR'}, "Object is not a generated die")
            return {'CANCELLED'}

        original_modifiers = set(body_obj.modifiers)

        mesh_cls_map = {
            "Tetrahedron": Tetrahedron,
            "D4Crystal": D4Crystal,
            "D4Shard": D4Shard,
            "Cube": Cube,
            "Octahedron": Octahedron,
            "Dodecahedron": Dodecahedron,
            "Icosahedron": Icosahedron,
            "D10Mesh": D10Mesh,
            "D100Mesh": D100Mesh,
        }

        mesh_cls = mesh_cls_map.get(die_type)
        if mesh_cls is None:
            self.report({'ERROR'}, f"Unknown dice type: {die_type}")
            return {'CANCELLED'}

        size = settings_values["size"]

        if die_type == "Tetrahedron":
            die = mesh_cls(body_obj.name, size, settings_values["number_center_offset"])
        elif die_type == "D4Crystal":
            die = mesh_cls(body_obj.name, size, settings_values["base_height"], settings_values["point_height"])
        elif die_type == "D4Shard":
            die = mesh_cls(
                body_obj.name,
                size,
                settings_values["top_point_height"],
                settings_values["bottom_point_height"],
                settings_values["number_v_offset"],
            )
        elif die_type in ("D10Mesh", "D100Mesh"):
            die = mesh_cls(body_obj.name, size, settings_values["height"], settings_values["number_v_offset"])
        else:
            die = mesh_cls(body_obj.name, size)

        die.dice_mesh = body_obj
        configure_dice_finish_modifier(
            body_obj,
            settings_values.get("dice_finish", "sharp"),
            settings_values.get("bumper_scale", 1),
        )

        font_path = validate_font_path(settings_values["font_path"]) if settings_values["font_path"] else ""
        custom_image_path = validate_svg_path(settings_values["custom_image_path"]) if settings_values["custom_image_path"] else ""
        settings_values["custom_image_path"] = custom_image_path

        new_numbers_obj = None

        if settings_values["add_numbers"]:
            try:
                if settings_values["number_indicator_type"] != NUMBER_IND_NONE:
                    new_numbers_obj = die.create_numbers(
                        context,
                        size,
                        settings_values["number_scale"],
                        settings_values["number_depth"],
                        font_path,
                        settings_values["one_offset"],
                        settings_values["number_indicator_type"],
                        settings_values["period_indicator_scale"],
                        settings_values["period_indicator_space"],
                        settings_values["bar_indicator_height"],
                        settings_values["bar_indicator_width"],
                        settings_values["bar_indicator_space"],
                        settings_values["center_bar"],
                        settings_values["custom_image_face"],
                        custom_image_path,
                        settings_values["custom_image_scale"],
                    )
                else:
                    new_numbers_obj = die.create_numbers(
                        context,
                        size,
                        settings_values["number_scale"],
                        settings_values["number_depth"],
                        font_path,
                        settings_values["one_offset"],
                        custom_image_face=settings_values["custom_image_face"],
                        custom_image_path=custom_image_path,
                        custom_image_scale=settings_values["custom_image_scale"],
                    )
            except Exception as exc:
                self.report({'ERROR'}, f"Failed to regenerate numbers: {exc}")
                return {'CANCELLED'}
        else:
            for mod in list(original_modifiers):
                if mod.type == 'BOOLEAN' and mod.name == 'boolean':
                    body_obj.modifiers.remove(mod)

            if numbers_obj and numbers_obj.name in bpy.data.objects:
                bpy.data.objects.remove(numbers_obj, do_unlink=True)

            if "dice_numbers_name" in body_obj:
                del body_obj["dice_numbers_name"]

            body_obj["dice_gen_type"] = die_type
            apply_settings(body_obj.dice_gen_settings, settings_values)
            return {'FINISHED'}

        if new_numbers_obj is not None:
            desired_name = numbers_obj.name if numbers_obj else "dice_numbers"
            if numbers_obj and numbers_obj.name in bpy.data.objects:
                numbers_obj.name = f"{numbers_obj.name}_old"

            for mod in list(original_modifiers):
                if mod.type == 'BOOLEAN' and mod.name == 'boolean':
                    body_obj.modifiers.remove(mod)

            new_numbers_obj["dice_body_name"] = body_obj.name
            new_numbers_obj["dice_gen_type"] = die_type
            apply_settings(new_numbers_obj.dice_gen_settings, settings_values)

            new_numbers_obj.name = desired_name
            body_obj["dice_numbers_name"] = new_numbers_obj.name

            if numbers_obj and numbers_obj.name in bpy.data.objects:
                bpy.data.objects.remove(numbers_obj, do_unlink=True)
        else:
            body_obj["dice_gen_type"] = die_type
            apply_settings(body_obj.dice_gen_settings, settings_values)

        return {'FINISHED'}


class OBJECT_PT_dice_gen(bpy.types.Panel):
    bl_label = "Dice Gen"
    bl_idname = "OBJECT_PT_dice_gen"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "object"

    @classmethod
    def poll(cls, context):
        return resolve_settings_owner(context.object) is not None

    def draw(self, context):
        layout = self.layout
        settings_owner = resolve_settings_owner(context.object)
        if settings_owner is None:
            layout.label(text="No dice settings found")
            return

        settings = settings_owner.dice_gen_settings

        col = layout.column()
        col.prop(settings, "font_path")
        col.prop(settings, "custom_image_path")
        col.prop(settings, "custom_image_face")
        col.prop(settings, "custom_image_scale")
        col.prop(settings, "number_scale")
        col.prop(settings, "number_depth")

        layout.separator()
        layout.operator("object.dice_gen_update", text="Regenerate Dice")


class MeshDiceAdd(Menu):
    """
    Dice menu under "Add Mesh"
    """

    bl_idname = 'VIEW3D_MT_mesh_dice_add'
    bl_label = 'Dice'

    def draw(self, context):
        layout = self.layout
        layout.operator_context = 'INVOKE_REGION_WIN'
        layout.operator('mesh.d4_add', text='D4 Tetrahedron')
        layout.operator('mesh.d4_crystal_add', text='D4 Crystal')
        layout.operator('mesh.d4_shard_add', text='D4 Shard')
        layout.operator('mesh.d6_add', text='D6 Cube')
        layout.operator('mesh.d8_add', text='D8 Octahedron')
        layout.operator('mesh.d10_add', text='D10 Trapezohedron')
        layout.operator('mesh.d100_add', text='D100 Trapezohedron')
        layout.operator('mesh.d12_add', text='D12 Dodecahedron')
        layout.operator('mesh.d20_add', text='D20 Icosahedron')


# Define "Extras" menu
def menu_func(self, context):
    layout = self.layout
    layout.operator_context = 'INVOKE_REGION_WIN'

    layout.separator()
    layout.menu('VIEW3D_MT_mesh_dice_add', text='Dice', icon='CUBE')


classes = [
    DiceGenSettings,
    MeshDiceAdd,
    D4Generator,
    D4CrystalGenerator,
    D4ShardGenerator,
    D6Generator,
    D8Generator,
    D10Generator,
    D100Generator,
    D12Generator,
    D20Generator,
    OBJECT_OT_dice_gen_update,
    OBJECT_PT_dice_gen
]


def register():
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)

    bpy.types.Object.dice_gen_settings = PointerProperty(type=DiceGenSettings)

    # Add "Dice" menu to the "Add Mesh" menu
    bpy.types.VIEW3D_MT_mesh_add.append(menu_func)


def unregister():
    # Remove "Dice" menu from the "Add Mesh" menu.
    bpy.types.VIEW3D_MT_mesh_add.remove(menu_func)

    del bpy.types.Object.dice_gen_settings

    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)
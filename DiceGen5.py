import math
import bpy
import os
import bmesh
from contextlib import contextmanager
from typing import List, Tuple, Optional, Dict, Any
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
THIRD_PI = math.pi / 3
QUARTER_PI = math.pi / 4
SIXTH_PI = math.pi / 6

# Empirically determined rotation angles for icosahedron number placement
# These values were derived using Blender's alignment trick for D20 dice
#
# TODO: Calculate these values analytically instead of using empirical measurements
# The analytical calculation should involve:
# 1. Computing the dihedral angle between icosahedron faces
# 2. Calculating the rotation needed to align numbers perpendicular to each face
# 3. Accounting for the golden ratio relationships inherent in icosahedron geometry
#
# For reference, the icosahedron has a dihedral angle of ~138.19° (acos(-sqrt(5)/3))
# and its geometry involves the golden ratio φ = (1 + sqrt(5)) / 2
ICOSAHEDRON_ROTATION_ANGLES = {
    'angle_1': 0.918438,  # ≈ 52.6°  - pitch angle for certain face orientations
    'angle_2': 2.82743,   # ≈ 162°   - yaw angle for face alignment
    'angle_3': 4.15881,   # ≈ 238.3° - roll angle for number orientation
    'angle_4': 0.314159,  # ≈ 18°    - small pitch adjustment (~π/10)
    'angle_5': 2.12437,   # ≈ 121.7° - complementary angle for opposite faces
    'angle_6': 4.06003,   # ≈ 232.7° - large pitch for inverted faces
    'angle_7': 2.22315,   # ≈ 127.4° - mid-range rotation angle
    'angle_8': 1.01722,   # ≈ 58.3°  - standard face orientation offset
}


def leg_b(leg_a: float, h: float) -> float:
    """
    Calculate the second leg of a right triangle given one leg and its height.

    Args:
        leg_a: Length of one leg of the right triangle
        h: Height of the triangle

    Returns:
        Length of the other leg
    """
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
    """
    Base class for polyhedral dice mesh generation.

    This class provides the foundation for creating different types of dice geometry
    and handles number placement on dice faces.

    Attributes:
        vertices: List of vertex coordinates for the mesh
        faces: List of face definitions (vertex indices)
        name: Name of the mesh
        dice_mesh: The created Blender mesh object
        base_font_scale: Base scaling factor for numbers on this die type
    """

    def __init__(self, name: str):
        """
        Initialize the mesh generator.

        Args:
            name: Name for the mesh object
        """
        self.vertices = None
        self.faces = None
        self.name = name
        self.dice_mesh = None
        self.base_font_scale = 1

    def create(self, context) -> bpy.types.Object:
        """
        Create the dice mesh in Blender.

        Args:
            context: Blender context

        Returns:
            The created mesh object
        """
        self.dice_mesh = create_mesh(context, self.vertices, self.faces, self.name)
        # reset transforms
        self.dice_mesh.matrix_world = Matrix()
        return self.dice_mesh

    def get_numbers(self) -> List[str]:
        """
        Get the list of numbers to place on the dice faces.

        Returns:
            List of number strings
        """
        return []

    def get_number_locations(self) -> List[Tuple[float, float, float]]:
        """
        Get the 3D positions for each number on the dice.

        Returns:
            List of (x, y, z) coordinate tuples
        """
        return []

    def get_number_rotations(self) -> List[Tuple[float, float, float]]:
        """
        Get the rotation angles for each number on the dice.

        Returns:
            List of (x, y, z) Euler angle tuples in radians
        """
        return []

    def create_numbers(self, context, size, number_scale, number_depth, font_path,
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
                                        center_bar, custom_image_face=custom_image_face,
                                        custom_image_path=custom_image_path, custom_image_scale=custom_image_scale)

        if numbers_object is not None:
            numbers_object.name = "dice_numbers"
            apply_boolean_modifier(self.dice_mesh, numbers_object)
            return numbers_object

        return None


class Tetrahedron(Mesh):
    """
    Tetrahedral dice (D4) mesh generator.

    Creates a regular tetrahedron with numbers placed on each face.
    The tetrahedron is oriented to stand on one face.
    """

    def __init__(self, name: str, size: float, number_center_offset: float, number_h_offset: float = 0.0, number_v_offset: float = 0.0):
        """
        Initialize a tetrahedron dice mesh.

        Args:
            name: Name for the mesh object
            size: Face-to-point size of the tetrahedron
            number_center_offset: How far numbers are offset from face centers (0=center, 1=vertex)
            number_h_offset: Horizontal offset for numbers on faces
            number_v_offset: Vertical offset for numbers on faces
        """
        super().__init__(name)
        self.size = size
        self.number_center_offset = number_center_offset
        self.number_h_offset = number_h_offset
        self.number_v_offset = number_v_offset

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

        # Calculate base positions using center offset
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

        # Apply horizontal and vertical offsets
        # For each face, we need to determine the local coordinate system
        if self.number_h_offset != 0.0 or self.number_v_offset != 0.0:
            c0 = CONSTANTS['tetrahedron']['c0'] / CONSTANTS['tetrahedron']['height'] * self.size
            scale_factor = c0 * 0.5  # Scale for offset application

            # Define face normal and up vectors for each of the 4 faces
            # Face 0: [0,1,2], Face 1: [1,0,3], Face 2: [2,3,0], Face 3: [3,2,1]
            face_info = [
                (0, vertices[2]),  # Numbers 0,1,2 - face 0
                (2, vertices[2]),  # Numbers 1,2 - face 2
                (3, vertices[2]),  # Numbers 2 - face 3
                (0, vertices[1]),  # Numbers 3,4,5 - face 0
                (1, vertices[1]),  # Numbers 4,5 - face 1
                (3, vertices[1]),  # Numbers 5 - face 3
                (0, vertices[0]),  # Numbers 6,7,8 - face 0
                (1, vertices[0]),  # Numbers 7,8 - face 1
                (2, vertices[0]),  # Numbers 8 - face 2
                (1, vertices[3]),  # Numbers 9,10,11 - face 1
                (2, vertices[3]),  # Numbers 10,11 - face 2
                (3, vertices[3]),  # Numbers 11 - face 3
            ]

            for i, (face_idx, target_vert) in enumerate(face_info):
                # Calculate face normal
                face = self.faces[face_idx]
                v0 = vertices[face[0]]
                v1 = vertices[face[1]]
                v2 = vertices[face[2]]
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = edge1.cross(edge2).normalized()

                # Direction from center to target vertex (this is our "up" direction)
                up_dir = (target_vert - centers[face_idx]).normalized()

                # Right direction is perpendicular to both normal and up
                right_dir = up_dir.cross(normal).normalized()

                # Apply offsets
                location_vectors[i] += right_dir * self.number_h_offset * scale_factor
                location_vectors[i] += up_dir * self.number_v_offset * scale_factor

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

    def __init__(self, name, size, base_height, top_point_height, bottom_point_height, number_h_offset: float = 0.0, number_v_offset: float = 0.0):
        super().__init__(name)
        self.size = size
        self.number_h_offset = number_h_offset
        self.number_v_offset = number_v_offset

        c0 = 0.5 * size
        c1 = 0.5 * base_height
        c2 = 0.5 * base_height + top_point_height
        c3 = -0.5 * base_height - bottom_point_height

        self.vertices = [(-c0, -c0, c1), (c0, -c0, c1), (c0, c0, c1), (-c0, c0, c1), (-c0, -c0, -c1), (c0, -c0, -c1),
                         (c0, c0, -c1), (-c0, c0, -c1), (0, 0, c2), (0, 0, c3)]
        self.faces = [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [0, 1, 8], [1, 2, 8], [2, 3, 8],
                      [3, 0, 8], [4, 5, 9], [5, 6, 9], [6, 7, 9], [7, 4, 9]]

        self.base_font_scale = 0.8

    def create(self, context):
        """Create the mesh and recalculate normals"""
        mesh_obj = super().create(context)
        # Recalculate normals to ensure they point outward
        bpy.ops.object.select_all(action='DESELECT')
        mesh_obj.select_set(True)
        bpy.context.view_layer.objects.active = mesh_obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode='OBJECT')
        return mesh_obj

    def get_numbers(self):
        return numbers(4)

    def get_number_locations(self):
        c0 = 0.5 * self.size
        h = self.number_h_offset * c0
        v = self.number_v_offset * c0
        # Apply offsets in each face's local coordinate system (all faces are on XY plane, rotated)
        return [(c0, h, v), (h, c0, v), (h, -c0, v), (-c0, h, v)]

    def get_number_rotations(self):
        return [(HALF_PI, 0, HALF_PI), (HALF_PI, 0, HALF_PI * 2), (HALF_PI, 0, 0), (HALF_PI, 0, HALF_PI * 3)]


class CustomCrystal(Mesh):
    """
    Custom crystal dice mesh generator.

    Creates a crystal-shaped die with a square base and pyramidal points on top and bottom.
    Supports any even number of faces (4, 6, 8, 10, 12, etc.) by placing numbers on the
    square sides of the die.
    """

    def __init__(self, name, size, num_faces, base_height, top_point_height, bottom_point_height, number_h_offset: float = 0.0, number_v_offset: float = 0.0):
        super().__init__(name)
        self.num_faces = num_faces
        self.size = size
        self.base_height = base_height
        self.top_point_height = top_point_height
        self.bottom_point_height = bottom_point_height
        self.number_h_offset = number_h_offset
        self.number_v_offset = number_v_offset

        # Calculate the number of sides on the base polygon
        # For a crystal, we have top pyramid + bottom pyramid + sides
        # If we want N total faces and the sides are where numbers go,
        # we need N sides (each side is where a number goes)
        self.num_sides = num_faces

        c0 = 0.5 * size
        c1 = 0.5 * base_height
        c2 = 0.5 * base_height + top_point_height
        c3 = -0.5 * base_height - bottom_point_height

        # Create vertices for a regular polygon base
        angle_step = 2 * math.pi / self.num_sides
        base_top_vertices = []
        base_bottom_vertices = []

        for i in range(self.num_sides):
            angle = i * angle_step
            x = c0 * math.cos(angle)
            y = c0 * math.sin(angle)
            base_top_vertices.append((x, y, c1))
            base_bottom_vertices.append((x, y, -c1))

        # Apex vertices
        top_apex = (0, 0, c2)
        bottom_apex = (0, 0, c3)

        # Combine all vertices
        self.vertices = base_top_vertices + base_bottom_vertices + [top_apex, bottom_apex]

        # Create faces
        faces = []

        # Side faces (quads connecting top and bottom base)
        for i in range(self.num_sides):
            next_i = (i + 1) % self.num_sides
            faces.append([i, next_i, next_i + self.num_sides, i + self.num_sides])

        # Top pyramid faces
        top_apex_idx = len(self.vertices) - 2
        for i in range(self.num_sides):
            next_i = (i + 1) % self.num_sides
            faces.append([i, next_i, top_apex_idx])

        # Bottom pyramid faces
        bottom_apex_idx = len(self.vertices) - 1
        for i in range(self.num_sides):
            next_i = (i + 1) % self.num_sides
            faces.append([i + self.num_sides, bottom_apex_idx, next_i + self.num_sides])

        self.faces = faces
        self.base_font_scale = 0.8

    def create(self, context):
        """Create the mesh and recalculate normals"""
        mesh_obj = super().create(context)
        # Recalculate normals to ensure they point outward
        bpy.ops.object.select_all(action='DESELECT')
        mesh_obj.select_set(True)
        bpy.context.view_layer.objects.active = mesh_obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode='OBJECT')
        return mesh_obj

    def get_numbers(self):
        return numbers(self.num_faces)

    def get_number_locations(self):
        # Place numbers on the center of each side face
        # For a regular polygon with vertices at radius c0, the distance to the middle
        # of an edge (apothem) is c0 * cos(π / num_sides)
        c0 = 0.5 * self.size
        angle_step = 2 * math.pi / self.num_sides
        # Calculate apothem - distance from center to middle of edge
        apothem = c0 * math.cos(math.pi / self.num_sides)
        h = self.number_h_offset * apothem
        v = self.number_v_offset * apothem
        locations = []

        for i in range(self.num_sides):
            angle = (i + 0.5) * angle_step  # Center of the side
            # Position numbers at the face surface
            # h moves tangent to the circle (perpendicular to radial direction)
            # v moves vertically in Z
            x = apothem * math.cos(angle) + h * (-math.sin(angle))
            y = apothem * math.sin(angle) + h * math.cos(angle)
            locations.append((x, y, v))

        return locations

    def get_number_rotations(self):
        # Rotate numbers to face outward from each side
        angle_step = 2 * math.pi / self.num_sides
        rotations = []

        for i in range(self.num_sides):
            angle = (i + 0.5) * angle_step
            rotations.append((HALF_PI, 0, angle + HALF_PI))

        return rotations


class CustomShard(Mesh):
    """
    Custom shard dice mesh generator.

    Creates a shard-shaped die with a regular polygon base and pyramidal points
    on top and bottom. Numbers are placed on the bottom pyramid faces.
    Supports various face counts (4, 6, 8, 10, 12, etc.).
    """

    def __init__(self, name, size, num_faces, top_point_height, bottom_point_height, number_v_offset, number_h_offset: float = 0.0):
        super().__init__(name)
        self.num_faces = num_faces
        self.size = size
        self.number_v_offset = number_v_offset
        self.number_h_offset = number_h_offset
        self.bottom_point_height = bottom_point_height
        self.top_point_height = top_point_height

        # For a shard, numbers go on the bottom pyramid faces
        self.num_sides = num_faces

        # Calculate radius for regular polygon
        c0 = size / (2 * math.sin(math.pi / self.num_sides))
        c1 = top_point_height * c0
        c2 = bottom_point_height * c0

        # Create vertices for a regular polygon base at z=0
        angle_step = 2 * math.pi / self.num_sides
        base_vertices = []

        for i in range(self.num_sides):
            angle = i * angle_step
            x = c0 * math.cos(angle)
            y = c0 * math.sin(angle)
            base_vertices.append((x, y, 0))

        # Apex vertices
        top_apex = (0, 0, c1)
        bottom_apex = (0, 0, -c2)

        # Combine all vertices
        self.vertices = base_vertices + [top_apex, bottom_apex]

        # Create faces
        faces = []
        top_apex_idx = len(base_vertices)
        bottom_apex_idx = len(base_vertices) + 1

        # Top pyramid faces (wind counter-clockwise when viewed from outside)
        for i in range(self.num_sides):
            next_i = (i + 1) % self.num_sides
            faces.append([i, next_i, top_apex_idx])

        # Bottom pyramid faces (wind clockwise to keep normals pointing outward)
        for i in range(self.num_sides):
            next_i = (i + 1) % self.num_sides
            faces.append([i, bottom_apex_idx, next_i])

        self.faces = faces
        self.base_font_scale = 0.8

    def create(self, context):
        """Create the mesh and recalculate normals"""
        mesh_obj = super().create(context)
        # Recalculate normals to ensure they point outward
        bpy.ops.object.select_all(action='DESELECT')
        mesh_obj.select_set(True)
        bpy.context.view_layer.objects.active = mesh_obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode='OBJECT')
        return mesh_obj

    def get_numbers(self):
        return numbers(self.num_faces)

    def get_number_locations(self):
        # Calculate number positions on bottom pyramid faces
        # Each face is a triangle: vertex i, vertex i+1, bottom apex
        # We interpolate along the face from the edge midpoint toward the apex

        angle_step = 2 * math.pi / self.num_sides
        c0 = self.size / (2 * math.sin(math.pi / self.num_sides))  # vertex radius
        c_bottom = self.bottom_point_height * c0  # bottom apex depth
        h = self.number_h_offset * c0

        locations = []
        for i in range(self.num_sides):
            # Get the two base vertices that form this face
            angle1 = i * angle_step
            angle2 = (i + 1) * angle_step

            # Midpoint of the two base vertices (on the base edge at z=0)
            edge_mid_x = (c0 * math.cos(angle1) + c0 * math.cos(angle2)) / 2
            edge_mid_y = (c0 * math.sin(angle1) + c0 * math.sin(angle2)) / 2

            # Interpolate from edge midpoint (at z=0) to bottom apex (at 0,0,-c_bottom)
            # At offset=1: at edge midpoint, at offset=0: at apex
            # h moves tangent to the face (perpendicular to radial direction)
            face_angle = (angle1 + angle2) / 2
            x = edge_mid_x * self.number_v_offset + h * (-math.sin(face_angle))
            y = edge_mid_y * self.number_v_offset + h * math.cos(face_angle)
            z = -c_bottom * (1 - self.number_v_offset)

            locations.append((x, y, z))

        return locations

    def get_number_rotations(self):
        # Calculate rotation by using the actual face normal vectors
        # Each bottom face is: vertex i, bottom_apex, vertex i+1

        angle_step = 2 * math.pi / self.num_sides
        c0 = self.size / (2 * math.sin(math.pi / self.num_sides))  # vertex radius
        c_bottom = self.bottom_point_height * c0

        rotations = []
        for i in range(self.num_sides):
            # Get the three vertices of this bottom pyramid face
            angle1 = i * angle_step
            angle2 = (i + 1) * angle_step
            center_angle = (i + 0.5) * angle_step

            # Vertex positions
            v1 = Vector((c0 * math.cos(angle1), c0 * math.sin(angle1), 0))
            v2 = Vector((0, 0, -c_bottom))  # bottom apex
            v3 = Vector((c0 * math.cos(angle2), c0 * math.sin(angle2), 0))

            # Calculate tilt angle using the same method as D4Shard
            # This measures the angle between a vertical vector and the vector to the edge midpoint
            edge_center_x = (c0 * math.cos(angle1) + c0 * math.cos(angle2)) / 2
            edge_center_y = (c0 * math.sin(angle1) + c0 * math.sin(angle2)) / 2

            # Vector from origin to vertical (above origin)
            vertical_vec = Vector((0, 0, c_bottom))
            # Vector from origin to edge center
            edge_vec = Vector((edge_center_x, edge_center_y, c_bottom))

            # Calculate angle between these vectors and add 90 degrees
            tilt_angle = math.pi / 2 + vertical_vec.angle(edge_vec)

            # Calculate the "up" vector on the face
            # The "up" direction should point from the bottom apex toward the top edge
            edge_midpoint = (v1 + v3) / 2  # midpoint of base edge
            up_on_face = edge_midpoint - v2  # vector from apex to edge midpoint
            up_on_face = up_on_face.normalized()

            # We need to find what angle to rotate the number around the face normal
            # After tilting, we want the number's local Y axis to point "up" along the face
            # The "up" direction on a tilted pyramid face points from apex toward the edge

            # Now calculate the angle we need to rotate around Y axis (roll)
            # This aligns the number's orientation with the face's orientation
            # The angle is measured from the radial direction
            radial_2d = Vector((math.cos(center_angle), math.sin(center_angle), 0))

            # Project up_on_face onto the XY plane to get its horizontal component
            up_horizontal = Vector((up_on_face.x, up_on_face.y, 0)).normalized()

            # Calculate angle between radial direction and up_horizontal
            # This is the roll we need to apply
            cos_roll = radial_2d.dot(up_horizontal)
            sin_roll = radial_2d.x * up_horizontal.y - radial_2d.y * up_horizontal.x
            y_rotation = math.atan2(sin_roll, cos_roll)

            # Add -90 degrees rotation around the Z axis to align the number with the face
            # Plus 180 degrees to flip the number so it reads correctly (not backwards)
            z_rotation = center_angle - math.pi / 2 + math.pi

            rotations.append((tilt_angle, y_rotation, z_rotation))

        return rotations


class D4Shard(CustomShard):
    """
    D4 Shard dice - a thin wrapper around CustomShard with 4 faces.

    This reuses the CustomShard logic which already handles positioning and rotation correctly.
    """

    def __init__(self, name, size, top_point_height, bottom_point_height, number_v_offset, number_h_offset: float = 0.0):
        # Simply call CustomShard with num_faces=4
        super().__init__(name, size, 4, top_point_height, bottom_point_height, number_v_offset, number_h_offset)


class CustomBipyramid(Mesh):
    """
    Custom bipyramid dice mesh generator.

    Creates a bipyramid (double pyramid) die with a regular polygon base and pyramidal points
    on both top and bottom. Numbers are placed on both the top and bottom pyramid faces.
    num_faces represents the total number of faces on the die (must be even, e.g., 6, 8, 10, 12, etc.).
    A die with 8 faces has a square base (4 sides), with 4 top faces + 4 bottom faces = 8 total.
    """

    def __init__(self, name, size, num_faces, top_point_height, bottom_point_height, number_h_offset: float = 0.0, number_v_offset: float = 0.0):
        """
        Initialize a custom bipyramid mesh.

        Args:
            name: Name for the mesh object
            size: Edge length of the base polygon
            num_faces: Total number of faces on the die (must be even, as it equals 2*base_polygon_sides)
            top_point_height: Height of the top pyramid point (relative to base radius)
            bottom_point_height: Height of the bottom pyramid point (relative to base radius)
            number_h_offset: Horizontal offset for numbers on faces
            number_v_offset: Vertical offset for numbers on faces
        """
        super().__init__(name)
        # Enforce even face count with minimum 6
        self.num_faces = max(6, num_faces if num_faces % 2 == 0 else num_faces + 1)  # Total number of faces
        self.size = size
        self.number_h_offset = number_h_offset
        self.number_v_offset = number_v_offset
        self.top_point_height = top_point_height
        self.bottom_point_height = bottom_point_height

        # Number of base polygon sides = num_faces / 2
        self.num_sides = self.num_faces // 2

        # Calculate radius for regular polygon
        c0 = size / (2 * math.sin(math.pi / self.num_sides))
        c1 = top_point_height * c0
        c2 = bottom_point_height * c0

        # Create vertices for a regular polygon base at z=0
        angle_step = 2 * math.pi / self.num_sides
        base_vertices = []

        for i in range(self.num_sides):
            angle = i * angle_step
            x = c0 * math.cos(angle)
            y = c0 * math.sin(angle)
            base_vertices.append((x, y, 0))

        # Apex vertices
        top_apex = (0, 0, c1)
        bottom_apex = (0, 0, -c2)

        # Combine all vertices
        self.vertices = base_vertices + [top_apex, bottom_apex]

        # Create faces
        faces = []
        top_apex_idx = len(base_vertices)
        bottom_apex_idx = len(base_vertices) + 1

        # Top pyramid faces (wind counter-clockwise when viewed from outside)
        for i in range(self.num_sides):
            next_i = (i + 1) % self.num_sides
            faces.append([i, next_i, top_apex_idx])

        # Bottom pyramid faces (wind clockwise to keep normals pointing outward)
        for i in range(self.num_sides):
            next_i = (i + 1) % self.num_sides
            faces.append([i, bottom_apex_idx, next_i])

        self.faces = faces
        self.base_font_scale = 0.8

    def create(self, context):
        """Create the mesh and recalculate normals"""
        mesh_obj = super().create(context)
        # Recalculate normals to ensure they point outward
        bpy.ops.object.select_all(action='DESELECT')
        mesh_obj.select_set(True)
        bpy.context.view_layer.objects.active = mesh_obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode='OBJECT')
        return mesh_obj

    def get_numbers(self):
        return numbers(self.num_faces)  # num_faces now represents total face count

    def get_number_locations(self):
        # Calculate number positions on both top and bottom pyramid faces using rotation matrices
        angle_step = 2 * math.pi / self.num_sides
        c0 = self.size / (2 * math.sin(math.pi / self.num_sides))  # vertex radius
        c_top = self.top_point_height * c0
        c_bottom = self.bottom_point_height * c0

        # Get rotations to derive local coordinate systems
        rotations = self.get_number_rotations()

        locations = []

        # Scale factors for offsets
        h_scale = self.number_h_offset * self.size * 0.5
        v_scale = self.number_v_offset * self.size * 0.5

        # Top pyramid faces
        for i in range(self.num_sides):
            # Get the two base vertices that form this face
            angle1 = i * angle_step
            angle2 = (i + 1) * angle_step

            # Midpoint of the two base vertices (on the base edge at z=0)
            edge_mid_x = (c0 * math.cos(angle1) + c0 * math.cos(angle2)) / 2
            edge_mid_y = (c0 * math.sin(angle1) + c0 * math.sin(angle2)) / 2

            # For top pyramid: lerp from edge midpoint toward top apex
            # Default position is 1/3 up from base edge to apex (looks good visually)
            lerp_factor = 0.33
            base_pos = Vector((
                edge_mid_x * (1 - lerp_factor),
                edge_mid_y * (1 - lerp_factor),
                c_top * lerp_factor
            ))

            # Use rotation matrix to get local coordinate system
            rot = rotations[i]
            euler = Euler(rot, 'XYZ')
            rot_matrix = euler.to_matrix()
            local_right = Vector((1, 0, 0))
            local_up = Vector((0, 1, 0))
            world_right = rot_matrix @ local_right
            world_up = rot_matrix @ local_up

            # Apply offsets in face-local coordinates
            pos = base_pos + world_right * h_scale + world_up * v_scale
            locations.append(tuple(pos))

        # Bottom pyramid faces
        for i in range(self.num_sides):
            # Get the two base vertices that form this face
            angle1 = i * angle_step
            angle2 = (i + 1) * angle_step

            # Midpoint of the two base vertices (on the base edge at z=0)
            edge_mid_x = (c0 * math.cos(angle1) + c0 * math.cos(angle2)) / 2
            edge_mid_y = (c0 * math.sin(angle1) + c0 * math.sin(angle2)) / 2

            # For bottom pyramid: lerp from edge midpoint toward bottom apex
            lerp_factor = 0.33
            base_pos = Vector((
                edge_mid_x * (1 - lerp_factor),
                edge_mid_y * (1 - lerp_factor),
                -c_bottom * lerp_factor
            ))

            # Use rotation matrix to get local coordinate system
            rot = rotations[self.num_sides + i]
            euler = Euler(rot, 'XYZ')
            rot_matrix = euler.to_matrix()
            local_right = Vector((1, 0, 0))
            local_up = Vector((0, 1, 0))
            world_right = rot_matrix @ local_right
            world_up = rot_matrix @ local_up

            # Apply offsets in face-local coordinates
            pos = base_pos + world_right * h_scale + world_up * v_scale
            locations.append(tuple(pos))

        return locations

    def get_number_rotations(self):
        # Calculate rotations for both top and bottom pyramid faces based on face normals
        angle_step = 2 * math.pi / self.num_sides
        c0 = self.size / (2 * math.sin(math.pi / self.num_sides))
        c_top = self.top_point_height * c0
        c_bottom = self.bottom_point_height * c0

        rotations = []

        # Precompute base vertices
        base_vertices = [Vector((c0 * math.cos(i * angle_step), c0 * math.sin(i * angle_step), 0)) for i in range(self.num_sides)]
        top_apex = Vector((0, 0, c_top))
        bottom_apex = Vector((0, 0, -c_bottom))

        def orientation_from_face(v0: Vector, v1: Vector, v2: Vector, apex: Vector, base_a: Vector, base_b: Vector) -> Tuple[float, float, float]:
            # Outward normal from face winding
            normal = (v1 - v0).cross(v2 - v0).normalized()

            # Up direction: project apex->edge_mid onto the face plane to keep numbers upright on the face
            edge_mid = (base_a + base_b) / 2
            up_hint = edge_mid - apex
            up_proj = (up_hint - normal * up_hint.dot(normal)).normalized()

            # Right-handed basis: X=right, Y=up, Z=normal
            right = up_proj.cross(normal).normalized()
            face_up = normal.cross(right).normalized()

            rot_matrix = Matrix((right, face_up, normal)).transposed()
            euler = rot_matrix.to_euler('XYZ')
            return (euler.x, euler.y, euler.z)

        # Top pyramid faces (winding: base_i, base_next, apex)
        for i in range(self.num_sides):
            v0 = base_vertices[i]
            v1 = base_vertices[(i + 1) % self.num_sides]
            rotations.append(orientation_from_face(v0, v1, top_apex, top_apex, v0, v1))

        # Bottom pyramid faces (winding: base_i, apex, base_next) to keep normals outward
        for i in range(self.num_sides):
            v0 = base_vertices[i]
            v1 = bottom_apex
            v2 = base_vertices[(i + 1) % self.num_sides]
            rotations.append(orientation_from_face(v0, v1, v2, bottom_apex, v0, v2))

        return rotations


class Cube(Mesh):
    """
    Cubic dice (D6) mesh generator.

    Creates a regular cube (hexahedron) with numbers 1-6 placed on each face.
    Opposite faces sum to 7 following standard dice conventions.
    """

    def __init__(self, name: str, size: float, number_h_offset: float = 0.0, number_v_offset: float = 0.0):
        """
        Initialize a cube dice mesh.

        Args:
            name: Name for the mesh object
            size: Face-to-face size of the cube
            number_h_offset: Horizontal offset for numbers on faces
            number_v_offset: Vertical offset for numbers on faces
        """
        super().__init__(name)

        # Calculate the necessary constants
        self.v_coord_const = 0.5 * size
        self.number_h_offset = number_h_offset
        self.number_v_offset = number_v_offset
        s = self.v_coord_const

        # create the vertices and faces
        self.vertices = [(-s, -s, -s), (s, -s, -s), (s, s, -s), (-s, s, -s), (-s, -s, s), (s, -s, s), (s, s, s),
                         (-s, s, s)]
        self.faces = [[0, 3, 2, 1], [0, 1, 5, 4], [0, 4, 7, 3], [6, 5, 1, 2], [6, 2, 3, 7], [6, 7, 4, 5]]

    def get_numbers(self):
        return numbers(6)

    def get_number_locations(self):
        s = self.v_coord_const
        h = self.number_h_offset * s
        v = self.number_v_offset * s
        # Each tuple represents (x, y, z) position of number on each face
        # Offsets are applied in the face's local coordinate system:
        # Face 1 (front, -Y): h along X, v along Z
        # Face 2 (left, -X): h along Z, v along Y
        # Face 3 (top, +Z): h along X, v along Y
        # Face 4 (bottom, -Z): h along X, v along Y
        # Face 5 (right, +X): h along Z, v along Y
        # Face 6 (back, +Y): h along X, v along Z
        return [(h, -s, v), (-s, -h, v), (h, v, s), (h, v, -s), (s, h, v), (h, s, v)]

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

    def __init__(self, name, size, number_h_offset: float = 0.0, number_v_offset: float = 0.0):
        super().__init__(name)

        # calculate circumscribed sphere radius from inscribed sphere radius
        # diameter of the inscribed sphere is the face 2 face length of the octahedron
        self.circumscribed_r = (size * math.sqrt(3)) / 2
        self.number_h_offset = number_h_offset
        self.number_v_offset = number_v_offset
        s = self.circumscribed_r

        # create the vertices and faces
        self.vertices = [(s, 0, 0), (-s, 0, 0), (0, s, 0), (0, -s, 0), (0, 0, s), (0, 0, -s)]
        self.faces = [[4, 0, 2], [4, 2, 1], [4, 1, 3], [4, 3, 0], [5, 2, 0], [5, 1, 2], [5, 3, 1], [5, 0, 3]]

        self.base_font_scale = 0.7

    def get_numbers(self):
        return numbers(8)

    def get_number_locations(self):
        c = self.circumscribed_r / 3

        # Base positions at face centers
        base_positions = [
            Vector((c, c, c)),      # Face 0
            Vector((-c, c, c)),     # Face 1
            Vector((-c, -c, c)),    # Face 2
            Vector((c, -c, c)),     # Face 3
            Vector((-c, c, -c)),    # Face 4
            Vector((c, c, -c)),     # Face 5
            Vector((c, -c, -c)),    # Face 6
            Vector((-c, -c, -c)),   # Face 7
        ]

        # Get the rotations to understand number orientation
        rotations = self.get_number_rotations()

        locations = []
        h_scale = self.number_h_offset * c
        v_scale = self.number_v_offset * c

        for i, rot in enumerate(rotations):
            # Create rotation matrix from Euler angles
            euler = Euler(rot, 'XYZ')
            rot_matrix = euler.to_matrix()

            # In the number's local space, X is right, Y is up (before rotation)
            # Apply rotation to get world-space directions
            local_right = Vector((1, 0, 0))
            local_up = Vector((0, 1, 0))

            world_right = rot_matrix @ local_right
            world_up = rot_matrix @ local_up

            # Start with base position and apply offsets
            pos = base_positions[i].copy()
            pos += world_right * h_scale + world_up * v_scale
            locations.append((pos.x, pos.y, pos.z))

        return locations

    def get_number_rotations(self):
        # dihedral angle / 2 - for tilting numbers to match face angle
        da = math.acos(-1 / 3)
        # Octahedron faces are organized as top pyramid (faces 0-3) and bottom pyramid (faces 4-7)
        # Each face is a triangle with a specific orientation that determines the z-rotation
        angles = [Euler((0, 0, 0), 'XYZ') for _ in range(8)]

        # Top pyramid faces - pointing upward (apex at vertex 4: (0,0,s))
        # Face 0: [4,0,2] - edge from +X to +Y
        angles[0].x = da / 2
        angles[0].z = math.pi * 3 / 4

        # Face 1: [4,2,1] - edge from +Y to -X
        angles[1].x = da / 2
        angles[1].z = math.pi * 5 / 4

        # Face 2: [4,1,3] - edge from -X to -Y
        angles[2].x = da / 2
        angles[2].z = math.pi * 7 / 4

        # Face 3: [4,3,0] - edge from -Y to +X
        angles[3].x = da / 2
        angles[3].z = math.pi * 1 / 4

        # Bottom pyramid faces - pointing downward (apex at vertex 5: (0,0,-s))
        # Face 4: [5,2,0] - number 5 (adjusting)
        angles[4].x = -math.pi + da / 2
        angles[4].z = math.pi * 1 / 4

        # Face 5: [5,1,2] - number 6 (CORRECT - don't change)
        angles[5].x = -math.pi + da / 2
        angles[5].z = math.pi * 7 / 4

        # Face 6: [5,3,1] - number 7 (CORRECT - don't change)
        angles[6].x = -math.pi + da / 2
        angles[6].z = math.pi * 5 / 4

        # Face 7: [5,0,3] - number 8 (adjusting)
        angles[7].x = -math.pi + da / 2
        angles[7].z = math.pi * 3 / 4

        return [(a.x, a.y, a.z) for a in angles]


class Dodecahedron(Mesh):

    def __init__(self, name, size, number_h_offset: float = 0.0, number_v_offset: float = 0.0):
        super().__init__(name)
        self.size = size
        self.number_h_offset = number_h_offset
        self.number_v_offset = number_v_offset

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
        # Numbers placed at dual polyhedron (icosahedron) vertices
        dual_e = self.size / 2 / CONSTANTS['icosahedron']['circumscribed_r']
        c0 = dual_e * CONSTANTS['icosahedron']['c0']
        c1 = dual_e * CONSTANTS['icosahedron']['c1']

        # Base positions at icosahedron vertices (dual of dodecahedron)
        base_positions = [
            Vector((c1, 0, c0)),
            Vector((0, c0, c1)),
            Vector((-c1, 0, c0)),
            Vector((0, -c0, c1)),
            Vector((c0, -c1, 0)),
            Vector((c0, c1, 0)),
            Vector((-c0, -c1, 0)),
            Vector((-c0, c1, 0)),
            Vector((0, c0, -c1)),
            Vector((c1, 0, -c0)),
            Vector((0, -c0, -c1)),
            Vector((-c1, 0, -c0)),
        ]

        # Get the rotations to understand number orientation
        rotations = self.get_number_rotations()

        locations = []
        h_scale = self.number_h_offset * c0
        v_scale = self.number_v_offset * c0

        for base_pos, rot in zip(base_positions, rotations):
            # Create rotation matrix from Euler angles
            euler = Euler(rot, 'XYZ')
            rot_matrix = euler.to_matrix()

            # In the number's local space, X is right, Y is up (before rotation)
            local_right = Vector((1, 0, 0))
            local_up = Vector((0, 1, 0))

            world_right = rot_matrix @ local_right
            world_up = rot_matrix @ local_up

            # Apply offsets in the number's coordinate system
            pos = base_pos.copy() + world_right * h_scale + world_up * v_scale
            locations.append((pos.x, pos.y, pos.z))

        return locations

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

    def __init__(self, name, size, number_h_offset: float = 0.0, number_v_offset: float = 0.0):
        super().__init__(name)
        self.size = size
        self.number_h_offset = number_h_offset
        self.number_v_offset = number_v_offset

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
        # Numbers are placed at the dual polyhedron (dodecahedron) vertices
        dual_e = self.size / 2 / CONSTANTS['octahedron']['circumscribed_r']

        c0 = CONSTANTS['octahedron']['c0'] * dual_e
        c1 = CONSTANTS['octahedron']['c1'] * dual_e
        s = CONSTANTS['octahedron']['c2'] * dual_e

        # Base positions at dodecahedron vertices
        base_positions = [
            Vector((0, s, c1)),      # Face 0
            Vector((-c0, -c0, -c0)), # Face 1
            Vector((s, c1, 0)),      # Face 2
            Vector((s, -c1, 0)),     # Face 3
            Vector((-c0, -c0, c0)),  # Face 4
            Vector((c1, 0, -s)),     # Face 5
            Vector((-c0, c0, c0)),   # Face 6
            Vector((0, s, -c1)),     # Face 7
            Vector((c1, 0, s)),      # Face 8
            Vector((-c0, c0, -c0)),  # Face 9
            Vector((c0, -c0, c0)),   # Face 10
            Vector((-c1, 0, -s)),    # Face 11
            Vector((0, -s, c1)),     # Face 12
            Vector((c0, -c0, -c0)),  # Face 13
            Vector((-c1, 0, s)),     # Face 14
            Vector((c0, c0, -c0)),   # Face 15
            Vector((-s, c1, 0)),     # Face 16
            Vector((-s, -c1, 0)),    # Face 17
            Vector((c0, c0, c0)),    # Face 18
            Vector((0, -s, -c1))     # Face 19
        ]

        # Get the rotations to understand number orientation
        rotations = self.get_number_rotations()

        locations = []
        h_scale = self.number_h_offset * self.size / 6
        v_scale = self.number_v_offset * self.size / 6

        for base_pos, rot in zip(base_positions, rotations):
            # Create rotation matrix from Euler angles
            euler = Euler(rot, 'XYZ')
            rot_matrix = euler.to_matrix()

            # In the number's local space, X is right, Y is up (before rotation)
            local_right = Vector((1, 0, 0))
            local_up = Vector((0, 1, 0))

            world_right = rot_matrix @ local_right
            world_up = rot_matrix @ local_up

            # Apply offsets in the number's coordinate system
            pos = base_pos.copy() + world_right * h_scale + world_up * v_scale
            locations.append((pos.x, pos.y, pos.z))

        return locations

    def get_number_rotations(self):
        """
        Calculate rotation angles for number placement on icosahedron faces.

        Note: Some angles are empirically determined. See ICOSAHEDRON_ROTATION_ANGLES
        for details and the TODO about calculating them analytically.

        Returns:
            List of rotation tuples (x, y, z) for each face
        """
        angles = [Euler((0, 0, 0), 'XYZ') for _ in range(20)]

        dihedral_half = (math.pi - CONSTANTS['icosahedron']['dihedral_angle']) / 2

        angles[0].x = -dihedral_half

        angles[1].x = -ICOSAHEDRON_ROTATION_ANGLES['angle_1']
        angles[1].y = -ICOSAHEDRON_ROTATION_ANGLES['angle_2']
        angles[1].z = -ICOSAHEDRON_ROTATION_ANGLES['angle_3']

        angles[2].x = HALF_PI
        angles[2].y = 5 * SIXTH_PI
        angles[2].z = math.pi - dihedral_half

        angles[3].x = HALF_PI
        angles[3].y = -SIXTH_PI
        angles[3].z = dihedral_half

        angles[4].x = -ICOSAHEDRON_ROTATION_ANGLES['angle_1']
        angles[4].y = -ICOSAHEDRON_ROTATION_ANGLES['angle_4']
        angles[4].z = ICOSAHEDRON_ROTATION_ANGLES['angle_5']

        angles[5].x = HALF_PI
        angles[5].y = THIRD_PI
        angles[5].z = HALF_PI
        angles[5].rotate(Euler((0, dihedral_half, 0), 'XYZ'))

        angles[6].x = -ICOSAHEDRON_ROTATION_ANGLES['angle_1']
        angles[6].y = ICOSAHEDRON_ROTATION_ANGLES['angle_4']
        angles[6].z = ICOSAHEDRON_ROTATION_ANGLES['angle_8']

        angles[7].x = -dihedral_half
        angles[7].y = math.pi

        angles[8].x = -HALF_PI
        angles[8].y = -THIRD_PI
        angles[8].z = -HALF_PI
        angles[8].rotate(Euler((0, -dihedral_half, 0), 'XYZ'))

        angles[9].x = -ICOSAHEDRON_ROTATION_ANGLES['angle_6']
        angles[9].y = ICOSAHEDRON_ROTATION_ANGLES['angle_4']
        angles[9].z = -ICOSAHEDRON_ROTATION_ANGLES['angle_5']

        angles[10].x = -ICOSAHEDRON_ROTATION_ANGLES['angle_1']
        angles[10].y = ICOSAHEDRON_ROTATION_ANGLES['angle_4']
        angles[10].z = -ICOSAHEDRON_ROTATION_ANGLES['angle_5']

        angles[11].x = HALF_PI
        angles[11].y = -THIRD_PI
        angles[11].z = -HALF_PI
        angles[11].rotate(Euler((0, -dihedral_half, 0), 'XYZ'))

        angles[12].x = -dihedral_half
        angles[12].z = math.pi

        angles[13].x = ICOSAHEDRON_ROTATION_ANGLES['angle_7']
        angles[13].y = ICOSAHEDRON_ROTATION_ANGLES['angle_4']
        angles[13].z = ICOSAHEDRON_ROTATION_ANGLES['angle_8']

        angles[14].x = -HALF_PI
        angles[14].y = THIRD_PI
        angles[14].z = HALF_PI
        angles[14].rotate(Euler((0, dihedral_half, 0), 'XYZ'))

        angles[15].x = -ICOSAHEDRON_ROTATION_ANGLES['angle_1']
        angles[15].y = -ICOSAHEDRON_ROTATION_ANGLES['angle_2']
        angles[15].z = -ICOSAHEDRON_ROTATION_ANGLES['angle_8']

        angles[16].x = HALF_PI
        angles[16].y = 7 * SIXTH_PI
        angles[16].z = math.pi + dihedral_half

        angles[17].x = HALF_PI
        angles[17].y = SIXTH_PI
        angles[17].z = -dihedral_half

        angles[18].x = -ICOSAHEDRON_ROTATION_ANGLES['angle_1']
        angles[18].y = -ICOSAHEDRON_ROTATION_ANGLES['angle_4']
        angles[18].z = -ICOSAHEDRON_ROTATION_ANGLES['angle_8']

        angles[19].x = math.pi
        angles[19].z = 2 * THIRD_PI
        angles[19].rotate(Euler((-dihedral_half, 0, 0), 'XYZ'))

        return [(a.x, a.y, a.z) for a in angles]


class SquashedPentagonalTrapezohedron(Mesh):
    """
    Pentagonal trapezohedron mesh generator (base for D10 and D100).

    This shape has 10 kite-shaped faces and is the standard shape for d10 dice.
    The shape can be "squashed" along the vertical axis by adjusting the height parameter.
    """

    def __init__(self, name: str, size: float, height: float, number_v_offset: float, number_h_offset: float = 0.0):
        """
        Initialize a pentagonal trapezohedron mesh.

        Args:
            name: Name for the mesh object
            size: Face-to-face size of the die
            height: Height scaling factor (1.0 = regular, <1.0 = squashed, >1.0 = elongated)
            number_v_offset: Vertical offset for number placement (0=bottom, 1=top of face)
            number_h_offset: Horizontal offset for number placement
        """
        super().__init__(name)
        self.size = size
        self.height = height
        self.number_v_offset = number_v_offset
        self.number_h_offset = number_h_offset

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

        # Face vertex pairs for vertical lerp
        face_vertices_data = [
            (vectors[6], vectors[8]),  # Face 0
            (vectors[3], vectors[9]),  # Face 1
            (vectors[1], vectors[8]),  # Face 2
            (vectors[4], vectors[9]),  # Face 3
            (vectors[10], vectors[8]), # Face 4
            (vectors[11], vectors[9]), # Face 5
            (vectors[7], vectors[8]),  # Face 6
            (vectors[2], vectors[9]),  # Face 7
            (vectors[0], vectors[8]),  # Face 8
            (vectors[5], vectors[9])   # Face 9
        ]

        # Get the rotations to understand number orientation
        rotations = self.get_number_rotations()

        locations = []
        lerp_factor = self.number_v_offset
        h_scale = self.number_h_offset * self.size / 4
        v_scale = self.number_v_offset * self.size / 4

        for (v1, v2), rot in zip(face_vertices_data, rotations):
            # Base position from vertical offset (lerp between two opposite vertices)
            base_pos = v1.lerp(v2, lerp_factor)

            # Create rotation matrix from Euler angles
            euler = Euler(rot, 'XYZ')
            rot_matrix = euler.to_matrix()

            # In the number's local space, X is right, Y is up (before rotation)
            local_right = Vector((1, 0, 0))
            local_up = Vector((0, 1, 0))

            world_right = rot_matrix @ local_right
            world_up = rot_matrix @ local_up

            # Apply offsets in the number's coordinate system
            pos = base_pos + world_right * h_scale + world_up * v_scale
            locations.append((pos.x, pos.y, pos.z))

        return locations

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

    def __init__(self, name, size, height, number_v_offset, number_h_offset: float = 0.0):
        super().__init__(name, size, height, number_v_offset, number_h_offset)
        self.base_font_scale = 0.6

    def get_numbers(self):
        return [str((i + 1) % 10) for i in range(10)]


class D100Mesh(SquashedPentagonalTrapezohedron):

    def __init__(self, name, size, height, number_v_offset, number_h_offset: float = 0.0):
        super().__init__(name, size, height, number_v_offset, number_h_offset)
        self.base_font_scale = 0.45

    def get_numbers(self):
        return [f'{str((i + 1) % 10)}0' for i in range(10)]


class CustomTrapezohedron(Mesh):
    """
    Custom trapezohedron (d10-style) with independent top/bottom point heights.

    Supports any even face count (minimum 6 faces). Top/bottom heights scale the positive/negative Z halves independently.
    """

    def __init__(self, name: str, size: float, num_faces: int, height: float, number_v_offset: float, number_h_offset: float = 0.0):
        Mesh.__init__(self, name)
        # Ensure an even face count of at least 6 (>= triangular trapezohedron)
        self.num_faces = max(6, num_faces if num_faces % 2 == 0 else num_faces + 1)
        self.num_sides = self.num_faces // 2
        self.size = size
        self.height = height
        self.number_v_offset = number_v_offset
        self.number_h_offset = number_h_offset
        self.base_font_scale = 0.5

        def build_antiprism(n: int):
            step = 2 * math.pi / n
            half = step / 2.0
            r = 1.0 / (2.0 * math.sin(math.pi / n))
            lateral_sq = 1.0 - 2 * r * r * (1 - math.cos(math.pi / n))
            h = math.sqrt(max(lateral_sq, 1e-8))
            z_top = h / 2.0
            z_bot = -h / 2.0

            verts = []
            for i in range(n):
                ang = i * step
                verts.append((r * math.cos(ang), r * math.sin(ang), z_top))
            for i in range(n):
                ang = i * step + half
                verts.append((r * math.cos(ang), r * math.sin(ang), z_bot))

            faces = []
            faces.append(list(range(n)))  # top
            faces.append(list(range(2 * n - 1, n - 1, -1)))  # bottom
            for i in range(n):
                a = i
                b = n + i
                c = n + ((i - 1) % n)
                faces.append([a, b, c])
                d = (i + 1) % n
                faces.append([a, d, b])
            return verts, faces

        def dual_mesh(verts, faces):
            vectors = [Vector(v) for v in verts]
            dual_verts = []
            for f in faces:
                v0, v1, v2 = (vectors[f[0]], vectors[f[1]], vectors[f[2]])
                normal = (v1 - v0).cross(v2 - v0)
                if normal.length == 0:
                    dual_verts.append(Vector((0, 0, 0)))
                    continue
                plane_offset = normal.dot(v0)
                dual_verts.append(normal / plane_offset)

            dual_faces = []
            for vi, v in enumerate(vectors):
                adjacent = []
                for fi, f in enumerate(faces):
                    if vi in f:
                        centroid = sum((vectors[idx] for idx in f), Vector((0, 0, 0))) / len(f)
                        adjacent.append((fi, centroid))

                axis = v.normalized()
                ref = Vector((1, 0, 0)) if abs(axis.x) < 0.9 else Vector((0, 1, 0))
                tangent = axis.cross(ref).normalized()
                bitangent = axis.cross(tangent).normalized()

                def angle_of(item):
                    fi, cent = item
                    vec = (dual_verts[fi] - v).normalized()
                    x = vec.dot(tangent)
                    y = vec.dot(bitangent)
                    return math.atan2(y, x)

                adjacent.sort(key=angle_of)
                dual_faces.append([fi for fi, _ in adjacent])

            return dual_verts, dual_faces

        # Build dual of a uniform antiprism to get an accurate trapezohedron
        anti_verts, anti_faces = build_antiprism(self.num_sides)
        trap_verts, trap_faces = dual_mesh(anti_verts, anti_faces)

        # Scale XY to requested size, Z independently to requested point heights
        xs = [v.x for v in trap_verts]
        ys = [v.y for v in trap_verts]
        zs = [v.z for v in trap_verts]
        current_radius = max(max(abs(min(xs)), abs(max(xs))), max(abs(min(ys)), abs(max(ys))), 1e-6)
        current_top = max(zs)
        current_bottom = min(zs)

        target_radius = size * 0.5
        target_top = target_radius * height
        target_bottom = -target_radius * height

        scale_xy = target_radius / current_radius
        scale_z = (target_top - target_bottom) / (current_top - current_bottom)

        def scale_vert(v: Vector):
            return (v.x * scale_xy, v.y * scale_xy, (v.z - current_bottom) * scale_z + target_bottom)

        self.vertices = [scale_vert(v) for v in trap_verts]
        self.faces = trap_faces

    def _face_frames(self):
        """
        Build local frames (center, right, up, normal) for each face with consistent orientation.
        """
        vectors = [Vector(v) for v in self.vertices]
        frames = []

        for face in self.faces:
            verts = [vectors[idx] for idx in face]
            normal = (verts[1] - verts[0]).cross(verts[2] - verts[0])
            if normal.length == 0:
                normal = Vector((0, 0, 1))
            normal.normalize()

            center = sum(verts, Vector((0, 0, 0))) / len(verts)
            if normal.dot(center) < 0:
                normal = -normal

            # Up points toward the apex (top or bottom) projected onto the face
            apex = max(verts, key=lambda v: abs(v.z))
            up_dir = apex - center
            up_dir = up_dir - normal * up_dir.dot(normal)
            if up_dir.length < 1e-8:
                up_dir = Vector((0, 1, 0))
            up_dir.normalize()

            right = up_dir.cross(normal)
            if right.length < 1e-8:
                right = Vector((1, 0, 0))
            right.normalize()

            frames.append({
                'center': center,
                'normal': normal,
                'up': up_dir,
                'right': right,
            })

        return frames

    def get_numbers(self):
        return numbers(self.num_faces)

    def get_number_locations(self):
        frames = self._face_frames()
        h_scale = self.number_h_offset * self.size / 4.0
        v_scale = self.number_v_offset * self.size / 4.0
        locations = []

        for frame in frames:
            base_pos = frame['center']
            pos = base_pos + frame['right'] * h_scale + frame['up'] * v_scale
            locations.append((pos.x, pos.y, pos.z))

        return locations

    def get_number_rotations(self):
        rotations = []
        for frame in self._face_frames():
            rot_matrix = Matrix((frame['right'], frame['up'], frame['normal'])).transposed()
            euler = rot_matrix.to_euler('XYZ')
            rotations.append((euler.x, euler.y, euler.z))
        return rotations


def numbers(n: int) -> List[str]:
    """
    Generate a list of number strings from 1 to n.

    Args:
        n: The count of numbers to generate

    Returns:
        List of string numbers from "1" to str(n)
    """
    return [str(i + 1) for i in range(n)]


def set_origin(o: bpy.types.Object, v: Vector) -> None:
    """
    Set the origin of an object to a specific location.

    Args:
        o: The Blender object to modify
        v: The new origin location as a Vector
    """
    me = o.data
    mw = o.matrix_world
    current = o.location
    T = Matrix.Translation(current - v)
    me.transform(T)
    mw.translation = mw @ v


def _calculate_bounds(vertices) -> Optional[Tuple[float, float, float, float, float, float]]:
    """
    Calculate the bounding box of a mesh's vertices.

    Args:
        vertices: Iterator or collection of mesh vertices

    Returns:
        Tuple of (min_x, max_x, min_y, max_y, min_z, max_z) or None if no vertices
    """
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


def set_origin_center_bounds(o: bpy.types.Object) -> None:
    """
    Set an object's origin to the center of its bounding box.

    Args:
        o: The Blender object to modify
    """
    me = o.data
    bounds = _calculate_bounds(me.vertices)
    if bounds is None:
        return

    min_x, max_x, min_y, max_y, min_z, max_z = bounds
    set_origin(o, Vector(((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2)))


def set_origin_min_bounds(o: bpy.types.Object) -> None:
    """
    Set an object's origin to the bottom-left corner of its bounding box.

    Args:
        o: The Blender object to modify
    """
    me = o.data
    bounds = _calculate_bounds(me.vertices)
    if bounds is None:
        return

    min_x, _, min_y, _, min_z, max_z = bounds
    set_origin(o, Vector((min_x, min_y, (min_z + max_z) / 2)))


def create_mesh(context, vertices: List[Tuple[float, float, float]],
                faces: List[List[int]], name: str) -> bpy.types.Object:
    """
    Create a Blender mesh object from vertices and faces.

    Args:
        context: Blender context
        vertices: List of vertex coordinates as (x, y, z) tuples
        faces: List of face definitions (each face is a list of vertex indices)
        name: Name for the mesh

    Returns:
        The created Blender object
    """
    verts = [Vector(i) for i in vertices]

    # Blender can handle n-gons directly, no need for createPolys
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    return object_data_add(context, mesh, operator=None)


def ensure_material(name: str, base_color: Tuple[float, float, float, float]) -> bpy.types.Material:
    """
    Create or retrieve a material with the specified name and color.

    Args:
        name: Name of the material
        base_color: RGBA color tuple (values 0.0-1.0)

    Returns:
        The material object
    """
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    if material.node_tree:
        bsdf = material.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = base_color
    return material


def assign_material(obj: bpy.types.Object, material: bpy.types.Material) -> None:
    """
    Assign a material to an object, replacing any existing materials.

    Args:
        obj: The Blender object
        material: The material to assign
    """
    if obj.data.materials:
        obj.data.materials.clear()
    obj.data.materials.append(material)


def apply_transform(ob: bpy.types.Object, use_location: bool = False,
                    use_rotation: bool = False, use_scale: bool = False) -> None:
    """
    Apply transforms to an object at a low level without operators.

    Based on: https://blender.stackexchange.com/questions/159538/

    Args:
        ob: The object to transform
        use_location: Apply location transform
        use_rotation: Apply rotation transform
        use_scale: Apply scale transform
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


FONT_EXTENSIONS = (".ttf", ".otf")


def validate_font_path(filepath: str) -> str:
    """
    Validate that a font file path exists and has a valid extension.

    Args:
        filepath: Path to the font file

    Returns:
        The filepath if valid, empty string otherwise
    """
    if not filepath:
        return ''

    if not os.path.isfile(filepath):
        return ''

    if os.path.splitext(filepath)[1].lower() not in FONT_EXTENSIONS:
        return ''

    return filepath


def validate_svg_path(filepath: str) -> str:
    """
    Validate that an SVG file path exists and has the .svg extension.

    Args:
        filepath: Path to the SVG file

    Returns:
        The filepath if valid, empty string otherwise
    """
    if not filepath:
        return ''

    if not os.path.isfile(filepath):
        return ''

    if os.path.splitext(filepath)[1].lower() != '.svg':
        return ''

    return filepath


def validate_dice_parameters(size: float, number_depth: float, number_scale: float) -> Tuple[bool, str]:
    """
    Validate dice generation parameters to ensure they produce valid geometry.

    Args:
        size: The face-to-face size of the die
        number_depth: Depth of number engravings
        number_scale: Scale of the numbers

    Returns:
        Tuple of (is_valid, error_message). error_message is empty if valid.
    """
    if size <= 0:
        return False, "Dice size must be greater than 0"

    if number_depth < 0:
        return False, "Number depth cannot be negative"

    if number_depth >= size / 2:
        return False, f"Number depth ({number_depth}) is too large for dice size ({size}). Should be less than {size/2}"

    if number_scale <= 0:
        return False, "Number scale must be greater than 0"

    return True, ""


SETTINGS_ATTRS = [
    "size",
    "dice_finish",
    "bumper_scale",
    "font_path",
    "number_scale",
    "number_depth",
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
    "number_h_offset",
    "num_faces",
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


def get_font(filepath: str) -> bpy.types.VectorFont:
    """
    Load a font from a file path, falling back to Blender's default font if loading fails.

    Args:
        filepath: Path to the font file (TTF or OTF)

    Returns:
        The loaded font object
    """
    if filepath:
        try:
            bpy.data.fonts.load(filepath=filepath, check_existing=True)
            return next(filter(lambda x: x.filepath == filepath, bpy.data.fonts))
        except (RuntimeError, OSError) as e:
            print(f"Warning: Could not load font from '{filepath}': {e}. Using default font.")
        except StopIteration:
            print(f"Warning: Font loaded but not found in bpy.data.fonts: '{filepath}'. Using default font.")

    # Fall back to Blender's built-in font
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
    inset_amount = 0.4 * bumper_scale
    extrude_amount = inset_amount * (0.5 / 0.3)

    if inset_amount <= 0 and extrude_amount <= 0:
        return

    bm = bmesh.new()
    bm.from_mesh(mesh_data)

    if not bm.faces:
        bm.free()
        return

    for face in bm.faces:
        face.tag = True

    inset_result = bmesh.ops.inset_individual(
        bm,
        faces=list(bm.faces),
        thickness=inset_amount,
        depth=0.0,
        use_even_offset=True,
    )

    inset_faces = list(inset_result.get("faces", []))

    # After the inset operation, Blender reports the new rim faces in
    # the "faces" result, while the original faces remain as the inset
    # centers. We want the raised bumper on the rim, so operate on the
    # inset result directly instead of inverting the set.
    rim_faces = inset_faces

    if extrude_amount > 0 and rim_faces:
        bm.normal_update()
        rim_verts = set()

        for face in rim_faces:
            rim_verts.update(face.verts)

        extrude_result = bmesh.ops.extrude_face_region(bm, geom=rim_faces)
        extruded_geom = extrude_result.get("geom", [])
        extruded_verts = [
            ele for ele in extruded_geom
            if isinstance(ele, bmesh.types.BMVert) and ele not in rim_verts
        ]

        if extruded_verts:
            bm.normal_update()
            for vert in extruded_verts:
                if vert.normal.length > 0:
                    vert.co += vert.normal.normalized() * extrude_amount

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


def create_text_mesh(context, text: str, font_path: str, font_size: float,
                     name: str, extrude: float = 0) -> bpy.types.Object:
    """
    Create a mesh object from text using a font.

    Args:
        context: Blender context
        text: Text string to create
        font_path: Path to font file (TTF or OTF)
        font_size: Size of the font
        name: Name for the created object
        extrude: Extrusion depth for 3D text

    Returns:
        The created mesh object
    """
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


def add_period_indicator(context, mesh_object: bpy.types.Object, number: str,
                         font_path: str, font_size: float, number_depth: float,
                         period_indicator_scale: float, period_indicator_space: float) -> bpy.types.Object:
    """
    Add a period indicator to numbers 6 and 9 for orientation.

    Args:
        context: Blender context
        mesh_object: The number mesh to add indicator to
        number: The number string ('6' or '9')
        font_path: Path to font file
        font_size: Base font size
        number_depth: Depth of the number extrusion
        period_indicator_scale: Scale factor for the period
        period_indicator_space: Spacing between number and period

    Returns:
        The combined mesh object with period indicator
    """
    p_obj = create_text_mesh(context, '.', font_path, font_size * period_indicator_scale,
                            f'period_{number}', number_depth)

    # move origin of period to the bottom left corner of the mesh
    set_origin_min_bounds(p_obj)

    space = (1 / 20) * font_size * period_indicator_space

    # move period to the bottom right of the number
    p_obj.location = Vector((mesh_object.location.x + (mesh_object.dimensions.x / 2) + space,
                             mesh_object.location.y - (mesh_object.dimensions.y / 2), 0))

    # join the period to the number
    return join([mesh_object, p_obj])


def add_bar_indicator(context, mesh_object: bpy.types.Object, font_size: float,
                      number_depth: float, bar_indicator_height: float,
                      bar_indicator_width: float, bar_indicator_space: float,
                      center_bar: bool) -> bpy.types.Object:
    """
    Add a bar indicator to numbers 6 and 9 for orientation.

    Args:
        context: Blender context
        mesh_object: The number mesh to add indicator to
        font_size: Base font size
        number_depth: Depth of the number extrusion
        bar_indicator_height: Height scale of the bar
        bar_indicator_width: Width scale of the bar
        bar_indicator_space: Spacing between number and bar
        center_bar: Whether to center-align the bar with the number

    Returns:
        The combined mesh object with bar indicator
    """
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

    return mesh_object


def create_numbers(context, numbers, locations, rotations, font_path, font_size, number_depth, number_indicator_type,
                   period_indicator_scale, period_indicator_space, bar_indicator_height, bar_indicator_width,
                   bar_indicator_space, center_bar, custom_image_face=0, custom_image_path='',
                   custom_image_scale=1):
    number_objs = []
    # create the number meshes
    for i in range(len(locations)):
        number_object = create_number(context, numbers[i], font_path, font_size, number_depth, locations[i],
                                      rotations[i], number_indicator_type, period_indicator_scale,
                                      period_indicator_space, bar_indicator_height, bar_indicator_width,
                                      bar_indicator_space, center_bar,
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
                  bar_indicator_space, center_bar, custom_image_face=0, custom_image_path='',
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
        if number in ('6', '9'):
            # Add orientation indicators for 6 and 9
            if number_indicator_type == NUMBER_IND_PERIOD:
                mesh_object = add_period_indicator(context, mesh_object, number, font_path, font_size,
                                                   number_depth, period_indicator_scale, period_indicator_space)
            elif number_indicator_type == NUMBER_IND_BAR:
                mesh_object = add_bar_indicator(context, mesh_object, font_size, number_depth,
                                                bar_indicator_height, bar_indicator_width,
                                                bar_indicator_space, center_bar)

    mesh_object.location.x = location[0]
    mesh_object.location.y = location[1]
    mesh_object.location.z = location[2]

    mesh_object.rotation_euler.x = rotation[0]
    mesh_object.rotation_euler.y = rotation[1]
    mesh_object.rotation_euler.z = rotation[2]

    for f in mesh_object.data.polygons:
        f.use_smooth = False

    return mesh_object


def execute_generator(op, context, mesh_cls, name: str, **kwargs) -> Dict[str, str]:
    """
    Main execution function for dice generation operators.

    This function coordinates the entire dice generation process:
    1. Validates input parameters
    2. Creates the dice geometry
    3. Applies finishing (bevel, bumpers, etc.)
    4. Generates and applies numbers
    5. Saves settings for regeneration

    Args:
        op: The operator instance containing user parameters
        context: Blender context
        mesh_cls: The Mesh subclass to instantiate (e.g., Cube, Icosahedron)
        name: Base name for the dice type
        **kwargs: Additional arguments to pass to the mesh_cls constructor

    Returns:
        Dictionary with 'FINISHED' status on success, 'CANCELLED' on failure
    """
    # Validate and sanitize file paths
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
                context, op.size, op.number_scale, op.number_depth, op.font_path,
                custom_image_face=op.custom_image_face, custom_image_path=op.custom_image_path,
                custom_image_scale=op.custom_image_scale
            )
        else:
            numbers_object = die.create_numbers(
                context, op.size, op.number_scale, op.number_depth, op.font_path,
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
def DiceSizeProperty(default: float):
    return FloatProperty(
        name='Dice Size',
        description='Size of the die (mm)',
        min=1,
        soft_min=1,
        max=100,
        soft_max=100,
        default=default
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
    description='Depth of the numbers on the die (mm)',
    min=0.1,
    soft_min=0.1,
    max=2,
    soft_max=2,
    default=0.75
)

FontPathProperty = StringProperty(
    name='Font',
    description='Number font (TTF or OTF)',
    maxlen=1024,
    subtype='FILE_PATH'
)

CustomImagePathProperty = StringProperty(
    name='Custom Image (SVG)',
    description='SVG file to engrave on a selected face',
    maxlen=1024,
    subtype='FILE_PATH'
)

def CustomImageFaceProperty(default=0):
    """
    Create a CustomImageFace property with configurable default.
    Pass the highest face number for the dice type to default to that face.
    """
    return IntProperty(
        name='Custom Image Face',
        description='1-based face index to replace with the custom image (0 disables the feature)',
        min=0,
        soft_min=0,
        default=default
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
        name="Dice Size",
        description="Size of the die (mm)",
        min=1,
        soft_min=1,
        max=100,
        soft_max=100,
        default=20
    )

    dice_finish: DiceFinishProperty()

    bumper_scale: BumperScaleProperty()

    font_path: FontPathProperty

    custom_image_path: CustomImagePathProperty

    custom_image_face: CustomImageFaceProperty(0)

    custom_image_scale: CustomImageScaleProperty

    number_scale: NumberScaleProperty

    number_depth: NumberDepthProperty


    add_numbers: AddNumbersProperty

    number_indicator_type: NumberIndicatorTypeProperty()

    period_indicator_scale: PeriodIndicatorScaleProperty

    period_indicator_space: PeriodIndicatorSpaceProperty

    bar_indicator_height: BarIndicatorHeightProperty

    bar_indicator_width: BarIndicatorWidthProperty

    bar_indicator_space: BarIndicatorSpaceProperty

    center_bar: CenterBarProperty

    number_v_offset: NumberVOffsetProperty(0.0)

    number_center_offset: FloatProperty(
        name='Number Center Offset',
        description='Distance of numbers from the center of a face',
        min=0.0,
        soft_min=0.0,
        max=1,
        soft_max=1,
        default=0.5
    )

    num_faces: IntProperty(
        name='Number of Faces',
        description='Number of faces on custom dice',
        min=3,
        soft_min=3,
        max=100,
        soft_max=40,
        default=6,
        step=1
    )

    base_height: FloatProperty(
        name='Base Height',
        description='Base height of the die (height of a face) (mm)',
        min=1,
        soft_min=1,
        max=100,
        soft_max=100,
        default=14
    )

    point_height: FloatProperty(
        name='Point Height',
        description='Point height of the die (mm)',
        min=1,
        soft_min=1,
        max=100,
        soft_max=100,
        default=7
    )

    top_point_height: FloatProperty(
        name='Top Point Height',
        description='Top point height of the die',
        min=0.25,
        soft_min=0.25,
        max=2,
        soft_max=2,
        default=7
    )

    bottom_point_height: FloatProperty(
        name='Bottom Point Height',
        description='Bottom point height of the die',
        min=0.25,
        soft_min=0.25,
        max=2.5,
        soft_max=2.5,
        default=7
    )

    height: FloatProperty(
        name='Dice Height',
        description='Height of the die',
        min=0.0,
        soft_min=0.0,
        max=100,
        soft_max=100,
        default=2 / 3
    )

    number_h_offset: FloatProperty(
        name='Number Horizontal Offset',
        description='Horizontal offset for number positioning on dice faces',
        min=-1.0,
        soft_min=-1.0,
        max=1.0,
        soft_max=1.0,
        default=0.0
    )


# ============================================================================
# OLD OPERATOR CLASSES - REMOVED
# All dice generation now uses DICE_OT_add_from_preset operator
# The individual operator classes (DiceGeneratorBase, D4Generator, D6Generator, etc.)
# have been removed as they are no longer used. The Add Mesh menu now calls
# DICE_OT_add_from_preset directly.
# Removed ~577 lines of duplicate code
# ============================================================================

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
            "CustomCrystal": CustomCrystal,
            "CustomShard": CustomShard,
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
            die = mesh_cls(body_obj.name, size, settings_values["base_height"], settings_values["top_point_height"], settings_values["bottom_point_height"])
        elif die_type == "CustomCrystal":
            die = mesh_cls(body_obj.name, size, settings_values["num_faces"], settings_values["base_height"], settings_values["top_point_height"], settings_values["bottom_point_height"])
        elif die_type == "D4Shard":
            die = mesh_cls(
                body_obj.name,
                size,
                settings_values["top_point_height"],
                settings_values["bottom_point_height"],
                settings_values["number_v_offset"],
            )
        elif die_type == "CustomShard":
            die = mesh_cls(
                body_obj.name,
                size,
                settings_values["num_faces"],
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
        """Draw the dice generation settings panel with organized property groups."""
        layout = self.layout
        settings_owner = resolve_settings_owner(context.object)
        if settings_owner is None:
            layout.label(text="No dice settings found")
            return

        settings = settings_owner.dice_gen_settings

        # Font Settings
        box = layout.box()
        box.label(text="Font", icon='FONT_DATA')
        box.prop(settings, "font_path")

        # Number Settings
        box = layout.box()
        box.label(text="Numbers", icon='OUTLINER_OB_FONT')
        box.prop(settings, "number_scale")
        box.prop(settings, "number_depth")

        # Custom Image Settings
        box = layout.box()
        box.label(text="Custom Image", icon='IMAGE_DATA')
        box.prop(settings, "custom_image_path")
        row = box.row()
        row.enabled = bool(settings.custom_image_path)
        row.prop(settings, "custom_image_face")
        row.prop(settings, "custom_image_scale")

        layout.separator()
        layout.operator("object.dice_gen_update", text="Regenerate Dice", icon='FILE_REFRESH')


class MeshDiceAdd(Menu):
    """
    Dice menu under "Add Mesh"
    """

    bl_idname = 'VIEW3D_MT_mesh_dice_add'
    bl_label = 'Dice'

    def draw(self, context):
        layout = self.layout
        layout.operator_context = 'INVOKE_REGION_WIN'

        # Use the sidebar operator for everything - single code path
        op = layout.operator('dicegen.add_from_preset', text='D4 Tetrahedron')
        op.dice_type = 'D4'

        op = layout.operator('dicegen.add_from_preset', text='D4 Crystal')
        op.dice_type = 'D4_CRYSTAL'

        op = layout.operator('dicegen.add_from_preset', text='D4 Shard')
        op.dice_type = 'D4_SHARD'

        op = layout.operator('dicegen.add_from_preset', text='D6 Cube')
        op.dice_type = 'D6'

        op = layout.operator('dicegen.add_from_preset', text='D8 Octahedron')
        op.dice_type = 'D8'

        op = layout.operator('dicegen.add_from_preset', text='D10 Trapezohedron')
        op.dice_type = 'D10'

        op = layout.operator('dicegen.add_from_preset', text='D100 Trapezohedron')
        op.dice_type = 'D100'

        op = layout.operator('dicegen.add_from_preset', text='D12 Dodecahedron')
        op.dice_type = 'D12'

        op = layout.operator('dicegen.add_from_preset', text='D20 Icosahedron')
        op.dice_type = 'D20'

        layout.separator()

        op = layout.operator('dicegen.add_from_preset', text='Custom Trapezohedron')
        op.dice_type = 'CUSTOM_TRAP'

        op = layout.operator('dicegen.add_from_preset', text='Custom Crystal')
        op.dice_type = 'CUSTOM_CRYSTAL'

        op = layout.operator('dicegen.add_from_preset', text='Custom Shard')
        op.dice_type = 'CUSTOM_SHARD'

        op = layout.operator('dicegen.add_from_preset', text='Custom Bipyramid')
        op.dice_type = 'CUSTOM_BIPYRAMID'


# Define "Extras" menu
def menu_func(self, context):
    layout = self.layout
    layout.operator_context = 'INVOKE_REGION_WIN'

    layout.separator()
    layout.menu('VIEW3D_MT_mesh_dice_add', text='Dice', icon='CUBE')


class DiceGenPresets(bpy.types.PropertyGroup):
    """PropertyGroup to store persistent dice generation settings in the scene"""

    dice_finish: DiceFinishProperty()
    bumper_scale: BumperScaleProperty()

    size: FloatProperty(
        name="Dice Size",
        description="Size of the die (mm)",
        min=1,
        soft_min=1,
        max=100,
        soft_max=100,
        default=20
    )

    add_numbers: AddNumbersProperty
    number_scale: NumberScaleProperty
    number_depth: NumberDepthProperty
    font_path: FontPathProperty

    number_indicator_type: NumberIndicatorTypeProperty()
    period_indicator_scale: PeriodIndicatorScaleProperty
    period_indicator_space: PeriodIndicatorSpaceProperty
    bar_indicator_height: BarIndicatorHeightProperty
    bar_indicator_width: BarIndicatorWidthProperty
    bar_indicator_space: BarIndicatorSpaceProperty
    center_bar: CenterBarProperty

    custom_image_path: CustomImagePathProperty
    custom_image_face: CustomImageFaceProperty(20)
    custom_image_scale: CustomImageScaleProperty

    number_center_offset: FloatProperty(
        name='Number Center Offset',
        description='Distance of numbers from the center of a face (D4 only)',
        min=0.0,
        soft_min=0.0,
        max=1,
        soft_max=1,
        default=0.5
    )

    num_faces: IntProperty(
        name='Number of Faces',
        description='Number of faces on custom dice',
        min=3,
        soft_min=3,
        max=100,
        soft_max=40,
        default=6,
        step=1
    )

    # Geometry-specific properties
    base_height: FloatProperty(
        name='Base Height',
        description='Base height of the die (D4 Crystal) (mm)',
        min=1,
        soft_min=1,
        max=100,
        soft_max=100,
        default=14
    )
    point_height: FloatProperty(
        name='Point Height',
        description='Point height of the die (D4 Crystal) (mm)',
        min=1,
        soft_min=1,
        max=100,
        soft_max=100,
        default=7
    )
    top_point_height: FloatProperty(
        name='Top Point Height',
        description='Top point height of the die',
        min=0.25,
        soft_min=0.25,
        max=100,
        soft_max=100,
        default=7
    )
    bottom_point_height: FloatProperty(
        name='Bottom Point Height',
        description='Bottom point height of the die',
        min=0.25,
        soft_min=0.25,
        max=100,
        soft_max=100,
        default=7
    )
    height: FloatProperty(
        name='Dice Height',
        description='Height of the die (D10/D100)',
        min=0.0,
        soft_min=0.0,
        max=100,
        soft_max=100,
        default=2 / 3
    )

    top_point_height_shard: FloatProperty(
        name='Top Point Height (Shard)',
        description='Top point height for shard dice (relative multiplier)',
        min=0.25,
        soft_min=0.25,
        max=2,
        soft_max=2,
        default=0.75
    )
    bottom_point_height_shard: FloatProperty(
        name='Bottom Point Height (Shard)',
        description='Bottom point height for shard dice (relative multiplier)',
        min=0.25,
        soft_min=0.25,
        max=2.5,
        soft_max=2.5,
        default=1.75
    )

    number_h_offset: FloatProperty(
        name='Number Horizontal Offset',
        description='Horizontal offset for number positioning on dice faces',
        min=-1.0,
        soft_min=-1.0,
        max=1.0,
        soft_max=1.0,
        default=0.0
    )

    number_v_offset_d4_shard: FloatProperty(
        name='Number Vertical Offset (D4 Shard)',
        description='Vertical offset of numbers for D4 Shard',
        min=0,
        soft_min=0,
        max=1,
        soft_max=1,
        default=0.75
    )
    number_v_offset: FloatProperty(
        name='Number Vertical Offset (D10/D100)',
        description='Vertical offset of numbers for D10 and D100',
        min=0,
        soft_min=0,
        max=1,
        soft_max=1,
        default=0.33
    )


class DICE_OT_add_from_preset(bpy.types.Operator):
    """Add a dice to the scene using preset settings"""
    bl_idname = 'dicegen.add_from_preset'
    bl_label = 'Add Dice'
    bl_options = {'REGISTER', 'UNDO'}

    dice_type: EnumProperty(
        name="Dice Type",
        items=[
            ('D4', 'D4', 'Tetrahedron'),
            ('D4_CRYSTAL', 'D4 Crystal', 'Crystal D4'),
            ('D4_SHARD', 'D4 Shard', 'Shard D4'),
            ('D6', 'D6', 'Cube'),
            ('D8', 'D8', 'Octahedron'),
            ('D10', 'D10', 'Trapezohedron'),
            ('D12', 'D12', 'Dodecahedron'),
            ('D20', 'D20', 'Icosahedron'),
            ('D100', 'D100', 'Trapezohedron'),
            ('CUSTOM_CRYSTAL', 'Custom Crystal', 'Custom Crystal Dice'),
            ('CUSTOM_SHARD', 'Custom Shard', 'Custom Shard Dice'),
            ('CUSTOM_BIPYRAMID', 'Custom Bipyramid', 'Custom Bipyramid Dice'),
            ('CUSTOM_TRAP', 'Custom Trapezohedron', 'Custom D10-style Trapezohedron'),
        ]
    )

    # Properties from presets - these will override preset values when set
    dice_finish: DiceFinishProperty()
    bumper_scale: BumperScaleProperty()
    size: FloatProperty(
        name="Dice Size",
        description="Size of the die (mm)",
        min=1,
        soft_min=1,
        max=100,
        soft_max=100,
        default=20
    )
    add_numbers: AddNumbersProperty
    number_scale: NumberScaleProperty
    number_depth: NumberDepthProperty
    font_path: FontPathProperty
    number_indicator_type: NumberIndicatorTypeProperty()
    period_indicator_scale: PeriodIndicatorScaleProperty
    period_indicator_space: PeriodIndicatorSpaceProperty
    bar_indicator_height: BarIndicatorHeightProperty
    bar_indicator_width: BarIndicatorWidthProperty
    bar_indicator_space: BarIndicatorSpaceProperty
    center_bar: CenterBarProperty
    custom_image_path: CustomImagePathProperty
    custom_image_face: IntProperty(
        name='Custom Image Face',
        description='1-based face index to replace with the custom image (0 disables the feature)',
        min=0,
        soft_min=0,
        default=0
    )
    custom_image_scale: CustomImageScaleProperty

    # Geometry-specific properties
    number_center_offset: FloatProperty(
        name='Number Center Offset',
        description='Distance of numbers from the center of a face (D4 only)',
        min=0.0,
        soft_min=0.0,
        max=1,
        soft_max=1,
        default=0.5
    )
    num_faces: IntProperty(
        name='Number of Faces',
        description='Number of faces on custom dice',
        min=3,
        soft_min=3,
        max=100,
        soft_max=40,
        default=6,
        step=2
    )
    base_height: FloatProperty(
        name='Base Height',
        description='Base height of the die (D4 Crystal) (mm)',
        min=1,
        soft_min=1,
        max=100,
        soft_max=100,
        default=14
    )
    point_height: FloatProperty(
        name='Point Height',
        description='Point height of the die (D4 Crystal) (mm)',
        min=1,
        soft_min=1,
        max=100,
        soft_max=100,
        default=7
    )
    top_point_height: FloatProperty(
        name='Top Point Height',
        description='Top point height of the die',
        min=0.25,
        soft_min=0.25,
        max=100,
        soft_max=100,
        default=3
    )
    bottom_point_height: FloatProperty(
        name='Bottom Point Height',
        description='Bottom point height of the die',
        min=0.25,
        soft_min=0.25,
        max=100,
        soft_max=100,
        default=3
    )
    height: FloatProperty(
        name='Dice Height',
        description='Height of the die (D10/D100)',
        min=0.0,
        soft_min=0.0,
        max=100,
        soft_max=100,
        default=2 / 3
    )
    number_v_offset: FloatProperty(
        name='Number Vertical Offset',
        description='Vertical offset of numbers on the dice faces',
        min=0,
        soft_min=0,
        max=1,
        soft_max=1,
        default=0.0
    )

    number_h_offset: FloatProperty(
        name='Number Horizontal Offset',
        description='Horizontal offset for number positioning on dice faces',
        min=-1.0,
        soft_min=-1.0,
        max=1.0,
        soft_max=1.0,
        default=0.0
    )

    def draw(self, context):
        """Draw the operator panel with only relevant properties for the selected dice type"""
        layout = self.layout

        # Define which properties are relevant for each dice type
        property_relevance = {
            'D4': ['size', 'number_center_offset', 'add_numbers', 'number_scale', 'number_depth',
                   'number_h_offset', 'number_v_offset', 'font_path', 'custom_image_path', 'custom_image_face', 'custom_image_scale'],
            'D4_CRYSTAL': ['size', 'base_height', 'top_point_height', 'bottom_point_height', 'add_numbers', 'number_scale',
                          'number_depth', 'number_h_offset', 'number_v_offset', 'font_path', 'custom_image_path', 'custom_image_face', 'custom_image_scale'],
            'D4_SHARD': ['size', 'top_point_height', 'bottom_point_height',
                        'add_numbers', 'number_scale', 'number_depth', 'number_h_offset', 'number_v_offset', 'font_path',
                        'custom_image_path', 'custom_image_face', 'custom_image_scale'],
            'D6': ['size', 'add_numbers', 'number_scale', 'number_depth', 'number_h_offset', 'number_v_offset', 'font_path',
                   'number_indicator_type', 'period_indicator_scale', 'period_indicator_space',
                   'bar_indicator_height', 'bar_indicator_width', 'bar_indicator_space', 'center_bar',
                   'custom_image_path', 'custom_image_face', 'custom_image_scale'],
            'D8': ['size', 'add_numbers', 'number_scale', 'number_depth', 'number_h_offset', 'number_v_offset', 'font_path',
                   'custom_image_path', 'custom_image_face', 'custom_image_scale'],
            'D10': ['size', 'height', 'add_numbers', 'number_scale', 'number_depth', 'number_h_offset', 'number_v_offset',
                   'font_path', 'custom_image_path', 'custom_image_face', 'custom_image_scale'],
            'D100': ['size', 'height', 'add_numbers', 'number_scale', 'number_depth', 'number_h_offset', 'number_v_offset',
                    'font_path', 'custom_image_path', 'custom_image_face', 'custom_image_scale'],
            'CUSTOM_TRAP': ['size', 'num_faces', 'height', 'add_numbers', 'number_scale',
                            'number_depth', 'number_h_offset', 'number_v_offset', 'font_path',
                            'number_indicator_type', 'period_indicator_scale', 'period_indicator_space',
                            'bar_indicator_height', 'bar_indicator_width', 'bar_indicator_space', 'center_bar',
                            'custom_image_path', 'custom_image_face', 'custom_image_scale'],
            'D12': ['size', 'add_numbers', 'number_scale', 'number_depth', 'number_h_offset', 'number_v_offset', 'font_path',
                    'custom_image_path', 'custom_image_face', 'custom_image_scale'],
            'D20': ['size', 'add_numbers', 'number_scale', 'number_depth', 'number_h_offset', 'number_v_offset', 'font_path',
                    'number_indicator_type', 'period_indicator_scale', 'period_indicator_space',
                    'custom_image_path', 'custom_image_face', 'custom_image_scale'],
            'CUSTOM_CRYSTAL': ['size', 'num_faces', 'base_height', 'top_point_height', 'bottom_point_height', 'add_numbers',
                              'number_scale', 'number_depth', 'number_h_offset', 'number_v_offset', 'font_path',
                              'number_indicator_type', 'period_indicator_scale', 'period_indicator_space',
                              'bar_indicator_height', 'bar_indicator_width', 'bar_indicator_space', 'center_bar',
                              'custom_image_path', 'custom_image_face', 'custom_image_scale'],
            'CUSTOM_SHARD': ['size', 'num_faces', 'top_point_height', 'bottom_point_height',
                            'add_numbers', 'number_scale', 'number_depth', 'number_h_offset', 'number_v_offset', 'font_path',
                            'number_indicator_type', 'period_indicator_scale', 'period_indicator_space',
                            'bar_indicator_height', 'bar_indicator_width', 'bar_indicator_space', 'center_bar',
                            'custom_image_path', 'custom_image_face', 'custom_image_scale'],
            'CUSTOM_BIPYRAMID': ['size', 'num_faces', 'top_point_height', 'bottom_point_height',
                                'add_numbers', 'number_scale', 'number_depth', 'number_h_offset', 'number_v_offset', 'font_path',
                                'number_indicator_type', 'period_indicator_scale', 'period_indicator_space',
                                'bar_indicator_height', 'bar_indicator_width', 'bar_indicator_space', 'center_bar',
                                'custom_image_path', 'custom_image_face', 'custom_image_scale'],
        }

        # Always show dice finish first
        layout.prop(self, "dice_finish")
        if self.dice_finish == "bumpers":
            layout.prop(self, "bumper_scale")

        # Get relevant properties for this dice type
        relevant_props = property_relevance.get(self.dice_type, [])

        # Clamp face counts for dice types that require even counts and a minimum of 6
        if self.dice_type in ['CUSTOM_TRAP', 'CUSTOM_BIPYRAMID']:
            if self.num_faces < 6:
                self.num_faces = 6
            if self.num_faces % 2 != 0:
                self.num_faces += 1

        # Draw properties in order
        for prop_name in relevant_props:
            if hasattr(self, prop_name):
                # Special handling for number indicator properties
                if prop_name == 'number_indicator_type':
                    supports_indicators = self.dice_type in ['D6', 'D8', 'D10', 'D12', 'D20', 'D100'] or (
                        self.dice_type == 'CUSTOM_TRAP' and self.num_faces >= 9) or (
                        self.dice_type in ['CUSTOM_CRYSTAL', 'CUSTOM_SHARD', 'CUSTOM_BIPYRAMID'] and self.num_faces >= 6)
                    if self.add_numbers and supports_indicators:
                        layout.prop(self, prop_name)
                elif prop_name in ['period_indicator_scale', 'period_indicator_space']:
                    supports_indicators = self.dice_type in ['D6', 'D8', 'D10', 'D12', 'D20', 'D100'] or (
                        self.dice_type == 'CUSTOM_TRAP' and self.num_faces >= 9) or (
                        self.dice_type in ['CUSTOM_CRYSTAL', 'CUSTOM_SHARD', 'CUSTOM_BIPYRAMID'] and self.num_faces >= 6)
                    if self.add_numbers and supports_indicators and self.number_indicator_type == 'period':
                        layout.prop(self, prop_name)
                elif prop_name in ['bar_indicator_height', 'bar_indicator_width', 'bar_indicator_space', 'center_bar']:
                    supports_indicators = self.dice_type in ['D6', 'D8', 'D10', 'D12', 'D20', 'D100'] or (
                        self.dice_type == 'CUSTOM_TRAP' and self.num_faces >= 9) or (
                        self.dice_type in ['CUSTOM_CRYSTAL', 'CUSTOM_SHARD', 'CUSTOM_BIPYRAMID'] and self.num_faces >= 6)
                    if self.add_numbers and supports_indicators and self.number_indicator_type == 'bar':
                        layout.prop(self, prop_name)
                else:
                    layout.prop(self, prop_name)

    def invoke(self, context, event):
        """Initialize operator properties from presets when invoked"""
        presets = context.scene.dicegen_presets

        # Copy values from presets
        self.dice_finish = presets.dice_finish
        self.bumper_scale = presets.bumper_scale
        self.size = presets.size
        self.add_numbers = presets.add_numbers
        self.number_scale = presets.number_scale
        self.number_depth = presets.number_depth
        self.font_path = presets.font_path
        self.number_indicator_type = presets.number_indicator_type
        self.period_indicator_scale = presets.period_indicator_scale
        self.period_indicator_space = presets.period_indicator_space
        self.bar_indicator_height = presets.bar_indicator_height
        self.bar_indicator_width = presets.bar_indicator_width
        self.bar_indicator_space = presets.bar_indicator_space
        self.center_bar = presets.center_bar
        self.custom_image_path = presets.custom_image_path
        self.custom_image_scale = presets.custom_image_scale
        self.number_center_offset = presets.number_center_offset
        self.base_height = presets.base_height
        self.point_height = presets.point_height
        self.top_point_height = presets.top_point_height
        self.bottom_point_height = presets.bottom_point_height
        self.height = presets.height
        self.num_faces = presets.num_faces
        self.number_h_offset = presets.number_h_offset

        # Default height for custom trapezohedron set to 1.0
        if self.dice_type == 'CUSTOM_TRAP':
            self.height = 1.0

        # Set number_v_offset based on dice type
        # Shard-type dice use 0.75, D10/D100 use 0.33, custom trapezohedron uses 0.0, all others use 0.0
        if self.dice_type in ['D4_SHARD', 'CUSTOM_SHARD']:
            self.number_v_offset = presets.number_v_offset_d4_shard  # 0.75
        elif self.dice_type in ['D10', 'D100']:
            self.number_v_offset = presets.number_v_offset  # 0.33
        elif self.dice_type == 'CUSTOM_TRAP':
            self.number_v_offset = 0.0
        else:
            self.number_v_offset = 0.0  # D4, D6, D8, D12, D20, D4_CRYSTAL, CUSTOM_CRYSTAL

        # Set point heights based on dice type
        # Crystal dice use absolute mm (7mm default), shards use relative multipliers (0.75, 1.75)
        if self.dice_type in ['D4_SHARD', 'CUSTOM_SHARD']:
            # For shards, use relative multiplier defaults
            self.top_point_height = presets.top_point_height_shard
            self.bottom_point_height = presets.bottom_point_height_shard
        elif self.dice_type == 'CUSTOM_BIPYRAMID':
            # Dedicated defaults for bipyramid (8 faces, symmetric points)
            self.num_faces = 8
            self.top_point_height = 2
            self.bottom_point_height = 2
        elif self.dice_type == 'CUSTOM_TRAP':
            # Defaults for custom trapezohedron
            self.num_faces = 10

        # Set custom_image_face based on dice type (highest face)
        dice_face_map = {
            'D4': 4,
            'D4_CRYSTAL': 4,
            'D4_SHARD': 4,
            'D6': 6,
            'D8': 8,
            'D10': 10,
            'D12': 12,
            'D20': 20,
            'D100': 10,  # D100 uses same faces as D10
            'CUSTOM_TRAP': self.num_faces,
            'CUSTOM_CRYSTAL': self.num_faces,
            'CUSTOM_SHARD': self.num_faces,
            'CUSTOM_BIPYRAMID': self.num_faces,  # num_faces now represents total face count
        }
        self.custom_image_face = dice_face_map.get(self.dice_type, presets.custom_image_face)

        # Enforce even face count and minimum 6 (total faces)
        if self.dice_type in ['CUSTOM_TRAP', 'CUSTOM_BIPYRAMID']:
            if self.num_faces < 6:
                self.num_faces = 6
            if self.num_faces % 2 != 0:
                self.num_faces += 1
        elif self.dice_type in ['CUSTOM_CRYSTAL', 'CUSTOM_SHARD']:
            if self.num_faces < 3:
                self.num_faces = 3

        return self.execute(context)

    def execute(self, context):
        # Map dice type to mesh class and generator params
        dice_map = {
            'D4': (Tetrahedron, 'd4', {'number_center_offset': self.number_center_offset, 'number_h_offset': self.number_h_offset, 'number_v_offset': self.number_v_offset}),
            'D4_CRYSTAL': (D4Crystal, 'd4Crystal', {'base_height': self.base_height, 'top_point_height': self.top_point_height, 'bottom_point_height': self.bottom_point_height, 'number_h_offset': self.number_h_offset, 'number_v_offset': self.number_v_offset}),
            'D4_SHARD': (D4Shard, 'd4Shard', {'top_point_height': self.top_point_height, 'bottom_point_height': self.bottom_point_height, 'number_v_offset': self.number_v_offset, 'number_h_offset': self.number_h_offset}),
            'CUSTOM_CRYSTAL': (CustomCrystal, 'customCrystal', {'num_faces': self.num_faces, 'base_height': self.base_height, 'top_point_height': self.top_point_height, 'bottom_point_height': self.bottom_point_height, 'number_h_offset': self.number_h_offset, 'number_v_offset': self.number_v_offset}),
            'CUSTOM_SHARD': (CustomShard, 'customShard', {'num_faces': self.num_faces, 'top_point_height': self.top_point_height, 'bottom_point_height': self.bottom_point_height, 'number_v_offset': self.number_v_offset, 'number_h_offset': self.number_h_offset}),
            'CUSTOM_BIPYRAMID': (CustomBipyramid, 'customBipyramid', {'num_faces': self.num_faces, 'top_point_height': self.top_point_height, 'bottom_point_height': self.bottom_point_height, 'number_h_offset': self.number_h_offset, 'number_v_offset': self.number_v_offset}),
            'D6': (Cube, 'd6', {'number_h_offset': self.number_h_offset, 'number_v_offset': self.number_v_offset}),
            'D8': (Octahedron, 'd8', {'number_h_offset': self.number_h_offset, 'number_v_offset': self.number_v_offset}),
            'D10': (D10Mesh, 'd10', {'height': self.height, 'number_v_offset': self.number_v_offset, 'number_h_offset': self.number_h_offset}),
            'D12': (Dodecahedron, 'd12', {'number_h_offset': self.number_h_offset, 'number_v_offset': self.number_v_offset}),
            'D20': (Icosahedron, 'd20', {'number_h_offset': self.number_h_offset, 'number_v_offset': self.number_v_offset}),
            'D100': (D100Mesh, 'd100', {'height': self.height, 'number_v_offset': self.number_v_offset, 'number_h_offset': self.number_h_offset}),
            'CUSTOM_TRAP': (CustomTrapezohedron, 'customTrapezohedron', {'num_faces': self.num_faces, 'height': self.height, 'number_v_offset': self.number_v_offset, 'number_h_offset': self.number_h_offset}),
        }

        if self.dice_type not in dice_map:
            self.report({'ERROR'}, f"Unknown dice type: {self.dice_type}")
            return {'CANCELLED'}

        mesh_class, name_prefix, extra_params = dice_map[self.dice_type]

        # Create the mesh
        mesh = mesh_class("dice_body", self.size, **extra_params)
        dice_obj = mesh.create(context)

        # Apply dice finish
        configure_dice_finish_modifier(dice_obj, self.dice_finish, self.bumper_scale)
        body_material = ensure_material("Dice Body", (0.95, 0.95, 0.9, 1))
        assign_material(dice_obj, body_material)

        # Collect settings for saving
        settings_values = {}
        for attr in SETTINGS_ATTRS:
            if hasattr(self, attr):
                settings_values[attr] = getattr(self, attr)

        numbers_object = None
        # Add numbers if enabled
        if self.add_numbers:
            number_indicator_type = NUMBER_IND_NONE
            supports_indicators = self.dice_type in ['D6', 'D8', 'D10', 'D12', 'D20', 'D100'] or (
                self.dice_type in ['CUSTOM_CRYSTAL', 'CUSTOM_SHARD', 'CUSTOM_BIPYRAMID'] and self.num_faces >= 6)

            if supports_indicators:
                number_indicator_type = self.number_indicator_type

            if number_indicator_type == NUMBER_IND_NONE:
                numbers_object = mesh.create_numbers(
                    context,
                    self.size,
                    self.number_scale,
                    self.number_depth,
                    self.font_path if self.font_path else '',
                    custom_image_face=self.custom_image_face,
                    custom_image_path=self.custom_image_path if self.custom_image_path else '',
                    custom_image_scale=self.custom_image_scale
                )
            else:
                numbers_object = mesh.create_numbers(
                    context,
                    self.size,
                    self.number_scale,
                    self.number_depth,
                    self.font_path if self.font_path else '',
                    number_indicator_type,
                    self.period_indicator_scale,
                    self.period_indicator_space,
                    self.bar_indicator_height,
                    self.bar_indicator_width,
                    self.bar_indicator_space,
                    self.center_bar,
                    custom_image_face=self.custom_image_face,
                    custom_image_path=self.custom_image_path if self.custom_image_path else '',
                    custom_image_scale=self.custom_image_scale
                )

        # Store metadata
        target_object = numbers_object or dice_obj
        target_object["dice_gen_type"] = mesh_class.__name__
        if numbers_object is not None:
            numbers_object["dice_body_name"] = dice_obj.name

        # Store settings on the object
        apply_settings(target_object.dice_gen_settings, settings_values)

        return {'FINISHED'}


class VIEW3D_PT_dice_gen_sidebar(bpy.types.Panel):
    """DiceGen panel in 3D viewport sidebar (N panel)"""
    bl_label = "DiceGen5"
    bl_idname = "VIEW3D_PT_dice_gen_sidebar"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'DiceGen5'

    def draw(self, context):
        layout = self.layout
        presets = context.scene.dicegen_presets

        # Dice Finish Settings
        box = layout.box()
        box.label(text="Dice Finish", icon='MOD_SUBSURF')
        box.prop(presets, "dice_finish")
        if presets.dice_finish == "bumpers":
            box.prop(presets, "bumper_scale")

        # Size Settings
        box = layout.box()
        box.label(text="Size", icon='EMPTY_ARROWS')
        box.prop(presets, "size")

        # Number Settings
        box = layout.box()
        box.label(text="Numbers", icon='OUTLINER_OB_FONT')
        box.prop(presets, "add_numbers")
        if presets.add_numbers:
            box.prop(presets, "number_scale")
            box.prop(presets, "number_depth")
            box.prop(presets, "font_path")

            # Number indicators (for D10/D100)
            box.label(text="Number Indicators (D10/D100):")
            box.prop(presets, "number_indicator_type")
            if presets.number_indicator_type == NUMBER_IND_PERIOD:
                box.prop(presets, "period_indicator_scale")
                box.prop(presets, "period_indicator_space")
            elif presets.number_indicator_type == NUMBER_IND_BAR:
                box.prop(presets, "bar_indicator_height")
                box.prop(presets, "bar_indicator_width")
                box.prop(presets, "bar_indicator_space")
                box.prop(presets, "center_bar")

        # Custom Image Settings
        box = layout.box()
        box.label(text="Custom Image", icon='IMAGE_DATA')
        box.prop(presets, "custom_image_path")
        if presets.custom_image_path:
            box.prop(presets, "custom_image_scale")
            box.label(text="(Image will appear on highest face)", icon='INFO')

        # Dice Type Buttons
        layout.separator()
        box = layout.box()
        box.label(text="Add Dice to Scene", icon='CUBE')

        col = box.column(align=True)
        op = col.operator("dicegen.add_from_preset", text="D4 Tetrahedron")
        op.dice_type = 'D4'

        op = col.operator("dicegen.add_from_preset", text="D4 Crystal")
        op.dice_type = 'D4_CRYSTAL'

        op = col.operator("dicegen.add_from_preset", text="D4 Shard")
        op.dice_type = 'D4_SHARD'

        op = col.operator("dicegen.add_from_preset", text="D6 Cube")
        op.dice_type = 'D6'

        op = col.operator("dicegen.add_from_preset", text="D8 Octahedron")
        op.dice_type = 'D8'

        op = col.operator("dicegen.add_from_preset", text="D10 Trapezohedron")
        op.dice_type = 'D10'

        op = col.operator("dicegen.add_from_preset", text="D12 Dodecahedron")
        op.dice_type = 'D12'

        op = col.operator("dicegen.add_from_preset", text="D20 Icosahedron")
        op.dice_type = 'D20'

        op = col.operator("dicegen.add_from_preset", text="D100 Trapezohedron")
        op.dice_type = 'D100'

        col.separator()

        op = col.operator("dicegen.add_from_preset", text="Custom Crystal")
        op.dice_type = 'CUSTOM_CRYSTAL'

        op = col.operator("dicegen.add_from_preset", text="Custom Shard")
        op.dice_type = 'CUSTOM_SHARD'

        op = col.operator("dicegen.add_from_preset", text="Custom Bipyramid")
        op.dice_type = 'CUSTOM_BIPYRAMID'

        op = col.operator("dicegen.add_from_preset", text="Custom Trapezohedron")
        op.dice_type = 'CUSTOM_TRAP'


classes = [
    DiceGenSettings,
    DiceGenPresets,
    MeshDiceAdd,
    # Old individual operators removed - Add Mesh menu now uses DICE_OT_add_from_preset
    # D4Generator,
    # D4CrystalGenerator,
    # D4ShardGenerator,
    # CustomCrystalGenerator,
    # CustomShardGenerator,
    # D6Generator,
    # D8Generator,
    # D10Generator,
    # D100Generator,
    # D12Generator,
    # D20Generator,
    OBJECT_OT_dice_gen_update,
    OBJECT_PT_dice_gen,
    DICE_OT_add_from_preset,  # Single unified operator for all dice generation
    VIEW3D_PT_dice_gen_sidebar
]


def register():
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)

    bpy.types.Object.dice_gen_settings = PointerProperty(type=DiceGenSettings)
    bpy.types.Scene.dicegen_presets = PointerProperty(type=DiceGenPresets)

    # Add "Dice" menu to the "Add Mesh" menu
    bpy.types.VIEW3D_MT_mesh_add.append(menu_func)


def unregister():
    # Remove "Dice" menu from the "Add Mesh" menu.
    bpy.types.VIEW3D_MT_mesh_add.remove(menu_func)

    del bpy.types.Object.dice_gen_settings
    del bpy.types.Scene.dicegen_presets

    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)

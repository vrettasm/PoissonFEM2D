import numpy as np
from math import isclose
from typing import Callable
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# Public interface.
__all__ = ["Mesh2D", "PoissonFEM2D"]


class Mesh2D(object):
    """
    Description:

    This class creates a Mesh2D object by constructing a 2D rectangular mesh.
    """

    # Use slots to reduce the memory footprint of the class.
    __slots__ = ('x_min', 'x_max', 'y_min', 'y_max', 'nx', 'ny',
                 'nodes', 'elements')

    def __init__(self, x_min: float = 0.0, x_max: float = 1.0, y_min: float = 0.0,
                 y_max: float = 1.0, nx: int = 10, ny: int = 10):
        """
        Default initializer.

        :param x_min (float): Minimum x-coordinate.

        :param x_max (float): Maximum x-coordinate.

        :param y_min (float): Minimum y-coordinate.

        :param y_max (float): Maximum y-coordinate.

        :param nx (int): Number of divisions along the x-axis.

        :param ny (int): Number of divisions along the y-axis.
        """

        # Make sure all the input values are finite numbers.
        if all(np.isfinite([x_min, x_max, y_min, y_max, nx, ny])):

            # Quick sanity check for the limit values of the x-axis.
            if x_min < x_max:
                self.x_min = float(x_min)
                self.x_max = float(x_max)
            else:
                raise ValueError(f"{self.__class__.__name__}: "
                                 "x_min should be less than x_max.")
            # _end_if_

            # Quick sanity check for the limit values of the y-axis.
            if y_min < y_max:
                self.y_min = float(y_min)
                self.y_max = float(y_max)
            else:
                raise ValueError(f"{self.__class__.__name__}: "
                                 "y_min should be less than y_max.")
            # _end_if_

            # Quick sanity check for the number of division on the x-axis.
            if nx > 0:
                self.nx = int(nx)
            else:
                raise ValueError(f"{self.__class__.__name__}: "
                                 "Number of divisions on x-axis should be positive.")
            # _end_if_

            # Quick sanity check for the number of division on the y-axis.
            if ny > 0:
                self.ny = int(ny)
            else:
                raise ValueError(f"{self.__class__.__name__}: "
                                 "Number of divisions on y-axis should be positive.")
            # _end_if_

        else:
            raise RuntimeError(f"{self.__class__.__name__}: "
                               "Something is wrong with the input values.")
        # _end_if_
        
        # Generate and store the grid nodes.
        self.nodes = self.create_nodes()
        
        # Generate and store the elements (here triangles).
        self.elements = self.create_elements()
    # _end_def_

    def create_nodes(self):
        """
        Create the nodes for the mesh [x_min, x_max] x [y_min, y_max].
        """

        # Create the x-space.
        x_space = np.linspace(self.x_min, self.x_max, self.nx + 1)

        # Create the y-space.
        y_space = np.linspace(self.y_min, self.y_max, self.ny + 1)

        # Use meshgrid to create the nodes.
        return np.array(np.meshgrid(x_space, y_space),
                        dtype=float).T.reshape(-1, 2)
    # _end_def_

    def create_elements(self):
        """
        Create the triangular elements for the mesh.
        """

        # Define the elements list.
        elements = []

        # Local copy of the append method.
        # (for faster calling in the double loop)
        elements_append = elements.append

        # Number of nodes on x-axis.
        node_count_x = self.nx + 1
        
        for j in range(self.ny):
            for i in range(self.nx):
                # Calculate the indices of the corners.
                i1 = i + (j * node_count_x)
                i2 = i1 + 1
                i3 = i1 + node_count_x
                i4 = i3 + 1
                
                # Append both triangles.
                elements_append([i1, i2, i3])
                elements_append([i2, i4, i3])
            # _end_for_
        # _end_for_

        # Make sure it is returned as 'int'.
        return np.array(elements, dtype=int)
    # _end_def_

    @property
    def node_count(self) -> int:
        """
        Return the number of nodes in the mesh.
        """
        return self.nodes.shape[0]
    # _end_def_

    @property
    def element_count(self) -> int:
        """
        Return the number of elements in the mesh.
        """
        return self.elements.shape[0]
    # _end_def_

    def node_coordinates(self, node_index: int):
        """
        Get the coordinates of a specific node.
        
        :param node_index (int): the index of the
        node on the 2D mesh.
        """
        return self.nodes[node_index]
    # _end_def_

    def element_nodes(self, element_index: int):
        """
        Get the nodes of a specific element.
        
        :param element_index (int): the index of
        the element (triangle).
        """
        return self.elements[element_index]
    # _end_def_

    def element_coordinates(self, element_index: int):
        """
        Get the coordinates of the nodes of a specific element.
        
        :param element_index (int): the index of
        the element (triangle).
        """
        node_indices = self.element_nodes(element_index)
        return self.nodes[node_indices]
    # _end_def_
    
    def calculate_quality_metrics(self):
        """
        Checks the quality of the 2D mesh for ensuring
        accurate and reliable simulation results.
        
        :return (tuple): 1) the aspect ratios of max/min sides,
                         2) the minimum angles and 3) the areas.
        """
        
        # Return list to hold the ratios.
        aspect_ratios = []
        
        # Return list to hold the minimum angle.
        min_angles = []
        
        # Return list to hold the areas.
        areas = []
        
        for element_index in range(self.element_count):
            
            # Get the element coordinates.
            element_coordinates = self.element_coordinates(element_index)
            
            # Get the points of the triangle.
            p1, p2, p3 = element_coordinates

            # Calculate side lengths.
            a = np.linalg.norm(p2 - p1)
            b = np.linalg.norm(p3 - p2)
            c = np.linalg.norm(p1 - p3)
            
            # Extract the (x, y) coordinates from the nodes.
            x1, y1 = element_coordinates[0]
            x2, y2 = element_coordinates[1]
            x3, y3 = element_coordinates[2]

            # Area of the triangular element.
            area = 0.5 * np.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
            
            # Store it in the return list.
            areas.append(area)
            
            # Calculate aspect ratio.
            aspect_ratio = np.max([a, b, c]) / np.min([a, b, c])
            
            # Store it in the return list.
            aspect_ratios.append(aspect_ratio)
            
            # Calculate angles in radians.
            angle_A = np.arccos((b**2 + c**2 - a**2) / (2.0 * b * c))
            angle_B = np.arccos((a**2 + c**2 - b**2) / (2.0 * a * c))
            angle_C = np.arccos((a**2 + b**2 - c**2) / (2.0 * a * b))
            
            # Convert angles to degrees.
            angles = np.degrees([angle_A, angle_B, angle_C])
            
            # Sanity check.
            if not isclose(np.sum(angles), 180.0):
                print("WARNING: angles don't sum to 180.0!")
            # _end_if_
            
            # Store the min angle in the return list.
            min_angles.append(np.min(angles))
            
        return aspect_ratios, min_angles, areas
    # _end_def_

# _end_class_


class PoissonFEM2D:
    """
    Solves the homogeneous Poisson equation using the Finite Element
    Method (FEM) with Dirichlet boundary conditions on a rectangular
    domain [a, b] x [c, d].
    """

    # Use slots to reduce the memory footprint of the class.
    __slots__ = ('mesh', 'boundary_condition', 'source_term',
                 'stiffness_matrix', 'load_vector', '_solution')

    def __init__(self, mesh: Mesh2D, boundary_condition: Callable,
                 source_term: Callable):
        """
        Initializes the PoissonFEM2D solver.

        :param mesh (Mesh2D): The mesh representing the 2D domain.

        :param boundary_condition (Callable[[float, float], float]):
        A function that takes (x, y) coordinates and returns the boundary
        condition value at that point.

        :param source_term (Callable[[float, float], float]): A function
        that takes (x, y) coordinates and returns the source term value
        at that point.
        """

        # Get the 2D mesh.
        self.mesh = mesh

        # Make sure the boundary conditions function is callable.
        if callable(boundary_condition):
            self.boundary_condition = boundary_condition
        else:
            raise TypeError(f"{self.__class__.__name__}: "
                            "Boundary conditions function is not callable.")
        # _end_if_

        # Make sure the source term function is callable.
        if callable(source_term):
            self.source_term = source_term
        else:
            raise TypeError(f"{self.__class__.__name__}: "
                            "Source term function is not callable.")
        # _end_if_

        # Placeholder for the solution.
        self._solution = None

        # Assemble the system to be solved.
        # (All the hard job is done here!)
        self.stiffness_matrix, self.load_vector = self.assemble_system()
    # _end_def_
    
    def local_stiffness(self, element_coordinates:np.ndarray):
        """
        Calculate local stiffness matrix and area of the triangle.

        :param element_coordinates (np.ndarray): The coordinates
        of the nodes of the element.

        :return (tuple): A tuple containing the local stiffness matrix
        and the area of the element (triangle).
        """

        # Extract the (x, y) coordinates from the nodes.
        x1, y1 = element_coordinates[0]
        x2, y2 = element_coordinates[1]
        x3, y3 = element_coordinates[2]

        # Area of the triangular element.
        area = 0.5 * np.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

        # This one assumes piecewise linear elements.
        D = np.array([[(x3 - x2), (x1 - x3), (x2 - x1)],
                      [(y3 - y2), (y1 - y3), (y2 - y1)]], dtype=float)
        
        # Element (local) stiffness matrix.
        elem_stiff = D.T.dot(D)/(4.0*area)
        
        # Return both local stiffness and area.
        return elem_stiff, area
    # _end_def_

    def assemble_system(self):
        """
        Assembles the global stiffness matrix and load vector.
        """

        # Get the total number of nodes in the mesh.
        num_nodes = self.mesh.node_count

        # Preallocate the 'A' matrix (sparse).
        # Will be converted to CSR in solve().
        A = lil_matrix((num_nodes, num_nodes), dtype=float)

        # Preallocate the 'b' vector.
        b = np.zeros(num_nodes, dtype=float)

        for element_index in range(self.mesh.element_count):
            # Get the element coordinates.
            element_coordinates = self.mesh.element_coordinates(element_index)

            # Element local stiffness matrix.
            elem_stiff, area = self.local_stiffness(element_coordinates)

            # Get the nodes of the current element.
            element_nodes = self.mesh.element_nodes(element_index)

            # Calculate the element's mean coordinates.
            mean_x, mean_y = element_coordinates.mean(axis=0)

            # Assemble into global stiffness matrix.
            for i in range(3):
                for j in range(3):
                    A[element_nodes[i], element_nodes[j]] += elem_stiff[i, j]
                # _end_for_

                # Evaluate the source term at the mean coordinates
                # and add its contribution to the load vector.
                b[element_nodes[i]] += self.source_term(mean_x, mean_y) * area / 3.0
            # _end_for_

        # _end_for_

        # Go through all the nodes and apply boundary conditions.
        for node_index in range(num_nodes):
            # Get the x/y values from the node.
            x, y = self.mesh.nodes[node_index]

            # Boundary node flag.
            is_boundary_node = (x == self.mesh.x_min or
                                x == self.mesh.x_max or
                                y == self.mesh.y_min or
                                y == self.mesh.y_max)

            # Check if boundary node.
            if is_boundary_node:
                A[node_index, :] = 0.0
                A[node_index, node_index] = 1.0
                b[node_index] = self.boundary_condition(x, y)
            # _end_if_

        # _end_for_

        return A, b
    # _end_def_

    def solve(self):
        """
          Solve the sparse linear system: "A*u=b" where:

          A := stiffness matrix
          b := load vector
        """

        # Assign the solution to the local variable.
        self._solution = spsolve(self.stiffness_matrix.tocsr(),
                                 self.load_vector)
    # _end_def_

    @property
    def solution(self):
        """
          Get the solution of the system.
        """
        return self._solution
    # _end_def_


# _end_class_

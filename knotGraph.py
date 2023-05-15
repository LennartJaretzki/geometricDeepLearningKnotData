import planarDiagram as pd
import numpy as np
import networkx as nx

distance_activation = lambda x: np.float(1 - np.exp(-x*(np.log(2)/15)))

def graphRepresentation(knot:pd.Knot):
    knot.relableKnot()
    knotFaces = []
    for path in knot.paths():
        face = set(knot.facingPaths(path))
        if not face in knotFaces:
            knotFaces.append(face)

    def delta(path):
        first_crossing, second_crossing = knot.findCrossingWithPath(path)
        return 1. if first_crossing.pathGoesOver(path) ^ second_crossing.pathGoesOver(path) else -1.0
        # ^ = xor returns difference
    
    max_path = max(knot.paths()) + 1 # +1 bcs relable knot starts relabeling at 0 so plus one
    def distance(path):
        first_crossing, second_crossing = knot.findCrossingWithPath(path)
        first_crossing_path = (first_crossing.corresponceRotationPath(path, rotation=1) + first_crossing.corresponceRotationPath(path, rotation=-1))
        second_crossing_path = (second_crossing.corresponceRotationPath(path, rotation=1) + second_crossing.corresponceRotationPath(path, rotation=-1))
        dist = (abs(first_crossing_path - second_crossing_path )//2)%(max_path//2)
        return dist



    def getKnotFaces(path):
        return list(filter(lambda knotFace: path in knotFace, knotFaces))

    graph = nx.Graph()
    for knotFace in knotFaces:
        graph.add_nodes_from(
            [knotFaces.index(knotFace)],
            x=[
                float(sum([delta(path) for path in knotFace]))
            ]
        )

    for path in knot.paths():
        face1, face2 = getKnotFaces(path)
        graph.add_edges_from(
            [
                (knotFaces.index(face1), knotFaces.index(face2),
                {"edge_attr":
                    (
                        distance_activation(distance(path)),
                        delta(path),
                        # float(distance(path)%2)
                    )
                })
            ]
        )
    
    return graph

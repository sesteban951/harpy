#!/usr/bin/env python

##
#
# Use Drake's ModelVisualizer to show the model, collision geometries, inertias,
# etc. in MeshCat.
#
##

from pydrake.all import ModelVisualizer, StartMeshcat

urdf_path = "./models/urdf/harpy_planar_CIMPC.urdf"

meshcat = StartMeshcat()
visualizer = ModelVisualizer(meshcat=meshcat)
visualizer.parser().AddModels(urdf_path)

visualizer.Run()


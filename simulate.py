#!/usr/bin/env python

##
#
# Run a simple simulation of the robot
#
##

import numpy as np
from pydrake.all import *

# Simulation parameters
sim_time = 4.0  # seconds
realtime_rate = 1

model_file = "./models/urdf/harpy.urdf"

config = MultibodyPlantConfig()
config.time_step = 5e-3
config.discrete_contact_solver = "sap"

# Start meshcat
meshcat = StartMeshcat()

# Set up the Drake system diagram
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlant(config, builder)

# Create the harpy model
Parser(plant).AddModels(model_file)
# TODO: add distance constraints

# Add a flat ground with friction
plant.RegisterCollisionGeometry(
        plant.world_body(),
        RigidTransform(),
        HalfSpace(),
        "ground_collision",
        CoulombFriction(static_friction=1.0, dynamic_friction=1.0))

# Turn off gravity
#plant.mutable_gravity_field().set_gravity_vector([0, 0, 0])

plant.Finalize()
AddDefaultVisualization(builder, meshcat)
diagram = builder.Build()
diagram_context = diagram.CreateDefaultContext()
plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)

# Set the initial condition
q0 = np.array([1, 0, 0, 0,   # base orientation
               0, 0, 0.49,   # base position
               0, 0,         # thrusters
               0, 0, 0, 0,   # right leg
               0, 0, 0, 0])  # left leg
plant.SetPositions(plant_context, q0)

# Run the sim
simulator = Simulator(diagram, diagram_context)
simulator.Initialize()
simulator.set_target_realtime_rate(realtime_rate)

meshcat.StartRecording()
simulator.AdvanceTo(sim_time)
meshcat.PublishRecording()


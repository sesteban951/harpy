#!/usr/bin/env python

##
#
# Run a simple simulation of the robot
#
##

from pydrake.all import *

# Simulation parameters
sim_time = 2.0  # seconds
realtime_rate = 1

model_file = "./models/urdf/harpy.urdf"

config = MultibodyPlantConfig()
config.time_step = 1e-3

# Start meshcat
meshcat = StartMeshcat()

# Set up the Drake system diagram
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlant(config, builder)
Parser(plant).AddModels(model_file)
plant.Finalize()
AddDefaultVisualization(builder, meshcat)
diagram = builder.Build()
diagram_context = diagram.CreateDefaultContext()

# Set the initial condition

# Run the sim
simulator = Simulator(diagram, diagram_context)
simulator.Initialize()
simulator.set_target_realtime_rate(realtime_rate)

meshcat.StartRecording()
simulator.AdvanceTo(sim_time)
meshcat.PublishRecording()


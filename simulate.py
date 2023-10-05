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
realtime_rate = -1

model_file = "./models/urdf/harpy.urdf"

config = MultibodyPlantConfig()
config.time_step = 1e-2
config.discrete_contact_solver = "sap"
config.contact_model = "hydroelastic_with_fallback"

# Start meshcat
meshcat = StartMeshcat()

# Set up the Drake system diagram
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlant(config, builder)

# Create the harpy model
harpy = Parser(plant).AddModels(model_file)[0]
body_A = plant.GetBodyByName("FootLeftBall")
body_B = plant.GetBodyByName("TibiaLeftBall")
plant.AddDistanceConstraint(
    plant.GetBodyByName("FootLeftBall"), [0, 0, 0],
    plant.GetBodyByName("TibiaLeftBall"), [0, 0, 0],
    0.32)
plant.AddDistanceConstraint(
    plant.GetBodyByName("FootRightBall"), [0, 0, 0],
    plant.GetBodyByName("TibiaRightBall"), [0, 0, 0],
    0.32)

# Add a flat ground with friction
ground_props = ProximityProperties()
AddContactMaterial(
        friction=CoulombFriction(static_friction=0.5, dynamic_friction=0.5),
        dissipation=1.25,
        properties=ground_props)
AddCompliantHydroelasticPropertiesForHalfSpace(
        slab_thickness=1.0,
        hydroelastic_modulus=1e7,
        properties=ground_props)
plant.RegisterCollisionGeometry(
        plant.world_body(),
        RigidTransform(),
        HalfSpace(),
        "ground_collision",
        ground_props)

# Turn off gravity
#plant.mutable_gravity_field().set_gravity_vector([0, 0, 0])

# Implicit PD
Kp = 50 * np.ones(plant.num_actuators())
Kd = 5 * np.ones(plant.num_actuators())

actuator_indices = []
for i in range(plant.num_actuators()):
    actuator_index = JointActuatorIndex(i)
    actuator_indices.append(actuator_index)

for actuator_index, Kp, Kd in zip(actuator_indices, Kp, Kd):
    plant.get_joint_actuator(actuator_index).set_controller_gains(
        PdControllerGains(p=Kp, d=Kd)
    )

plant.Finalize()

# TODO: add simple PD controller
controller = builder.AddSystem(
        ConstantVectorSource(np.zeros(plant.num_actuators())))
builder.Connect(
        controller.get_output_port(),
        plant.get_actuation_input_port())

# Desired state sender
desired_state_sender = builder.AddSystem(
        ConstantVectorSource(np.zeros(16)))
builder.Connect(
        desired_state_sender.get_output_port(),
        plant.get_desired_state_input_port(harpy))

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

print("done")

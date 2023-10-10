#!/usr/bin/env python

##
#
# Run a simple simulation of the robot
#
##

import numpy as np
from pydrake.all import *
from planar_controller import PlanarRaibertController

# Simulation parameters
sim_time = 10.0  # seconds
realtime_rate = 1

model_file = "./models/urdf/harpy_planar.urdf"

config = MultibodyPlantConfig()
config.time_step = 1e-2
config.discrete_contact_solver = "sap"
config.contact_model = "hydroelastic_with_fallback"

# Start meshcat
meshcat = StartMeshcat()

# Set up the Drake system diagram
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlant(config, builder)

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

# Add the harpy model
harpy = Parser(plant).AddModels(model_file)[0]
plant.AddDistanceConstraint(
    plant.GetBodyByName("BallTarsusLeft"), [0, 0, 0],
    plant.GetBodyByName("BallFemurLeft"), [0, 0, 0],
    0.32)
plant.AddDistanceConstraint(
    plant.GetBodyByName("BallTarsusRight"), [0, 0, 0],
    plant.GetBodyByName("BallFemurRight"), [0, 0, 0],
    0.32)

# Disable gravity (for debugging)
plant.gravity_field().set_gravity_vector([0, 0, 0])

# Set up control strategy. The user-designed controller supplies nominal joint
# angles q_nom, nominal joint velocities v_nom, and a feed-forward torque tau_ff
# for each of the actuated joints. Torques to be applied on the robot are then
# computed as 
#
#   tau = tau_ff + Kp * (q - q_nom) + Kd * (v - v_nom)
#
# This is a rough imitation of a low-level motor control strategy that might
# run on the hardware.
Kp = 50 * np.ones(plant.num_actuators())
Kd = 5 * np.ones(plant.num_actuators())
actuator_indices = [JointActuatorIndex(i) for i in range(plant.num_actuators())]
for actuator_index, Kp, Kd in zip(actuator_indices, Kp, Kd):
    plant.get_joint_actuator(actuator_index).set_controller_gains(
        PdControllerGains(p=Kp, d=Kd)
    )
plant.Finalize()

# Add thrusters
left_thruster = builder.AddSystem(
        Propeller(plant.GetBodyByName("ThrusterLeft").index()))
right_thruster = builder.AddSystem(
        Propeller(plant.GetBodyByName("ThrusterRight").index()))

spatial_force_multiplexer = builder.AddSystem(  # combines forces of both props
        ExternallyAppliedSpatialForceMultiplexer(2))

builder.Connect(
        plant.get_body_poses_output_port(),
        left_thruster.get_body_poses_input_port())
builder.Connect(
        plant.get_body_poses_output_port(),
        right_thruster.get_body_poses_input_port())

builder.Connect(
        left_thruster.get_spatial_forces_output_port(),
        spatial_force_multiplexer.get_input_port(0))
builder.Connect(
        right_thruster.get_spatial_forces_output_port(),
        spatial_force_multiplexer.get_input_port(1))
builder.Connect(
        spatial_force_multiplexer.get_output_port(),
        plant.get_applied_spatial_force_input_port())

thruster_demux = builder.AddSystem(
    Demultiplexer(2, 1))
builder.Connect(
        thruster_demux.get_output_port(0),
        left_thruster.get_command_input_port())
builder.Connect(
        thruster_demux.get_output_port(1),
        right_thruster.get_command_input_port())

# Add the controller
controller = builder.AddSystem(
        PlanarRaibertController())
builder.Connect(
        plant.get_state_output_port(),
        controller.GetInputPort("x_hat"))
builder.Connect(
        controller.GetOutputPort("tau_ff"),
        plant.get_actuation_input_port())
builder.Connect(
        controller.GetOutputPort("x_nom"),
        plant.get_desired_state_input_port(harpy))
builder.Connect(
        controller.GetOutputPort("thrust"),
        thruster_demux.get_input_port())

AddDefaultVisualization(builder, meshcat)
diagram = builder.Build()
diagram_context = diagram.CreateDefaultContext()
plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)

# Set the initial condition
q0 = np.array([0, 0.515,           # base position
               0,       # base orientation
               0, 0,         # thrusters
               0, 0, 0, 0,   # right leg
               0, 0, 0, 0])  # left leg
plant.SetPositions(plant_context, q0)

# Initialize the sim
simulator = Simulator(diagram, diagram_context)
simulator.set_target_realtime_rate(realtime_rate)
simulator.Initialize()

# Run the sim
meshcat.StartRecording()
simulator.AdvanceTo(sim_time)
meshcat.PublishRecording()


#!/usr/bin/env python

##
#
# Run a simple simulation of the robot
#
##

import numpy as np
from pydrake.all import *
from planar_controller import PlanarRaibertController
from hlip_controller import HybridLIPController

from info_logging import InfoLogger

# Simulation parameters
sim_time = 10.0      # seconds
realtime_rate = 0   # speed of simulation relative to real time

# choose controller type: "raibert" or "hlip"
controller_type = "raibert"

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
        friction=CoulombFriction(static_friction=1.0, dynamic_friction=0.7),
        dissipation=0,
        properties=ground_props)
AddCompliantHydroelasticPropertiesForHalfSpace(
        slab_thickness=0.1,
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

# Add data logger for generic info
info_sys = builder.AddSystem(InfoLogger())

# Disable gravity (for debugging)
plant.gravity_field().set_gravity_vector([0, 0, -9.81])

# Set up control strategy. The user-designed controller supplies nominal joint
# angles q_nom, nominal joint velocities v_nom, and a feed-forward torque tau_ff
# for each of the actuated joints. Torques to be applied on the robot are then
# computed as 
#
#   tau = tau_ff + Kp * (q - q_nom) + Kd * (v - v_nom)
#
# This is a rough imitation of a low-level motor control strategy that might
# run on the hardware.
Kp = 450 * np.ones(plant.num_actuators())
Kd = 50 * np.ones(plant.num_actuators())
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

# Add the walking controller
if controller_type == "hlip":
    controller = builder.AddSystem(HybridLIPController())
elif controller_type == "raibert":
    controller = builder.AddSystem(PlanarRaibertController())

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

# Add the info logger
builder.Connect(plant.get_state_output_port(),
        	    info_sys.GetInputPort("state"))

# logger for state data
state_logger = LogVectorOutput(plant.get_state_output_port(), builder)
com_logger = LogVectorOutput(info_sys.GetOutputPort("com_state"), builder)
mom_logger = LogVectorOutput(info_sys.GetOutputPort("momentum"), builder)
base_logger = LogVectorOutput(info_sys.GetOutputPort("base_pos"), builder)
right_foot_logger = LogVectorOutput(info_sys.GetOutputPort("right_foot_pos"), builder)
left_foot_logger = LogVectorOutput(info_sys.GetOutputPort("left_foot_pos"), builder)

AddDefaultVisualization(builder, meshcat)
diagram = builder.Build()
diagram_context = diagram.CreateDefaultContext()
plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)

# Set the initial condition
q0 = np.array([0, 0.514,     # base position (514 mm default height)
               0,            # base orientation
               0, 0,         # thrusters
               0, 0, 0, 0,   # right leg
               0, 0, 0, 0])  # left leg
v0 = np.array([0.3,0,          # base velocity
               0,            # base angular velocity
               0,0,          # thrusters angular velocity
               0,0,0,0,      # right leg joint angular velocity
               0,0,0,0])     # left leg joint  angular velocity
plant.SetPositions(plant_context, q0)
plant.SetVelocities(plant_context, v0)

# Initialize the sim
simulator = Simulator(diagram, diagram_context)
simulator.set_target_realtime_rate(realtime_rate)
simulator.Initialize()

# Run the sim
meshcat.StartRecording()
simulator.AdvanceTo(sim_time)
meshcat.PublishRecording()

# Retrieve all the data from the loggers
state_data = state_logger.FindLog(diagram_context)
com_data = com_logger.FindLog(diagram_context)
mom_data = mom_logger.FindLog(diagram_context)
base_data = base_logger.FindLog(diagram_context)
right_foot_data = right_foot_logger.FindLog(diagram_context)
left_foot_data = left_foot_logger.FindLog(diagram_context)

time = state_data.sample_times()
state = state_data.data().transpose()
com = com_data.data().transpose()
mom = mom_data.data().transpose()
base = base_data.data().transpose(),
right = right_foot_data.data().transpose()
left = left_foot_data.data().transpose()

data_len = len(np.array(time).flatten())

# Save all data to text files in data folder
time = np.array(time).reshape((data_len, 1))
state = np.array(state).reshape((data_len, 26))
com = np.array(com).reshape((data_len, 6))
mom = np.array(mom).reshape((data_len, 6))
base = np.array(base).reshape((data_len, 3))
right = np.array(right).reshape((data_len, 3))
left = np.array(left).reshape((data_len, 3))

np.savetxt("data/time.txt", time)
np.savetxt("data/state.txt", state)
np.savetxt("data/com.txt", com)
np.savetxt("data/mom.txt", mom)
np.savetxt("data/base.txt", base)
np.savetxt("data/right.txt", right)
np.savetxt("data/left.txt", left)

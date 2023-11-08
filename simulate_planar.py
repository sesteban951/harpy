#!/usr/bin/env python

##
#
# Run a simple simulation of the robot
#
##

import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import *
from planar_controller import PlanarRaibertController

# Simulation parameters
sim_time = 2.0       # seconds
realtime_rate = 1    # speed of simulation relative to real time
plot_state = True   # plot the state data

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
plant.gravity_field().set_gravity_vector([0, 0, -3.7])

# Set up control strategy. The user-designed controller supplies nominal joint
# angles q_nom, nominal joint velocities v_nom, and a feed-forward torque tau_ff
# for each of the actuated joints. Torques to be applied on the robot are then
# computed as 
#
#   tau = tau_ff + Kp * (q - q_nom) + Kd * (v - v_nom)
#
# This is a rough imitation of a low-level motor control strategy that might
# run on the hardware.
Kp = 250 * np.ones(plant.num_actuators())
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

# logger for state data
logger = LogVectorOutput(plant.get_state_output_port(), builder)

AddDefaultVisualization(builder, meshcat)
diagram = builder.Build()
diagram_context = diagram.CreateDefaultContext()
plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)

# Set the initial condition
q0 = np.array([0, 0.5,     # base position (514 mm default height)
               0,            # base orientation
               0, 0,         # thrusters
               0, 0, 0, 0,   # right leg
               0, 0, 0, 0])  # left leg
v0 = np.array([0,0,          # base velocity
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

# plot the state data
if plot_state:
	# get the state log
	state_log = logger.FindLog(diagram_context)
	time = state_log.sample_times()
	data = state_log.data().transpose()

	# Parse the state data
	pos_base = data[:, 0:2]
	ang_base = data[:, 2]
	pos_thruster = data[:, 3:5]
	pos_right_leg = data[:, 5:9]
	pos_left_leg = data[:, 9:13]

	vel_base = data[:, 13:15]
	ang_vel_base = data[:, 15]
	vel_thruster = data[:, 16:18]
	vel_right_leg = data[:, 18:22]
	vel_left_leg = data[:, 22:26]

	# Create a figure with subplots
	fig, axs = plt.subplots(5, 2, sharex=True)

	# Base Position
	axs[0, 0].plot(time, pos_base)
	axs[0, 0].set_ylabel("Position [m]")
	axs[0, 0].legend(["x", "z"])
	axs[0, 0].set_title("Base Pos")
	axs[0, 0].grid(True)
	axs[0, 1].plot(time, vel_base)
	axs[0, 1].set_ylabel("Velocity [m/s]")
	axs[0, 1].legend(["x", "z"])
	axs[0, 1].set_title("Base Vel")
	axs[0, 1].grid(True)

	# Base Orientation
	axs[1, 0].plot(time, ang_base)
	axs[1, 0].set_ylabel("Orientation [rad]")
	axs[1, 0].set_title("Base Pitch Pos")
	axs[1, 0].grid(True)
	axs[1, 1].plot(time, ang_vel_base)
	axs[1, 1].set_ylabel("Angular Velocity [rad/s]")
	axs[1, 1].set_title("Base Pitch Vel")
	axs[1, 1].grid(True)

	# Thruster Orientation
	axs[2, 0].plot(time, pos_thruster)
	axs[2, 0].set_ylabel("Position [rad]")
	axs[2, 0].legend(["R", "L"])
	axs[2, 0].set_title("Thruster Pos")
	axs[2, 0].grid(True)
	axs[2, 1].plot(time, vel_thruster)
	axs[2, 1].set_ylabel("Velocity [rad/s]")
	axs[2, 1].legend(["R", "L"])
	axs[2, 1].set_title("Thruster Vel")
	axs[2, 1].grid(True)

	# Right Leg orietnation
	axs[3, 0].plot(time, pos_right_leg)
	axs[3, 0].set_ylabel("Position [rad]")
	axs[3, 0].legend(["q1", "q2", "q3", "q4"])
	axs[3, 0].set_title("Right Leg Pos")
	axs[3, 0].grid(True)
	axs[3, 1].plot(time, vel_right_leg)
	axs[3, 1].set_ylabel("Velocity [rad/s]")
	axs[3, 1].legend(["q1", "q2", "q3", "q4"])
	axs[3, 1].set_title("Right Leg Vel")
	axs[3, 1].grid(True)

	# Left leg orientation
	axs[4, 0].plot(time, pos_left_leg)
	axs[4, 0].set_xlabel("Time [s]")
	axs[4, 0].set_ylabel("Position [rad]")
	axs[4, 0].legend(["q1", "q2", "q3", "q4"])
	axs[4, 0].set_title("Left Leg Pos")
	axs[4, 0].grid(True)
	axs[4, 1].plot(time, vel_left_leg)
	axs[4, 1].set_xlabel("Time [s]")
	axs[4, 1].set_ylabel("Velocity [rad/s]")
	axs[4, 1].legend(["q1", "q2", "q3", "q4"])
	axs[4, 1].set_title("Left Leg Vel")
	axs[4, 1].grid(True)

	# Show the plot
	plt.show()
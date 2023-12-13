# Harpy Humanoid Robot
The purpose of this repo is to prototype different controllers for the Harpy walking/flying humanoid robot.

## Dependecies
There are few dependencies that you need to run this code. 

### Drake
This repo uses Drake which is a robotics simulation tool maintained by Toyota Research Institute. For this repo, we only use Drake python bindings. Refer to [drake install instructions](https://drake.mit.edu/installation.html) for how to install the Drake python bindings which entail building with `bazel`. 

### Inverse Dynamics Trajectory Optimization
Some controllers rely on a model predictive control (MPC) strategy. We use recent [CI-MPC](https://arxiv.org/abs/2309.01813) work from Kurtz et. al. as base for our MPC control schemes. Specifically, we use the python implementation, `pyidto`. Refer to the [idto](https://github.com/ToyotaResearchInstitute/idto) repo for instructions on how to build the `pyidto` python bindings with `bazel`.

## Setup
The simplest way to include python bindings in the the python scripts is by simply appending the location of the compiled `pyidto` python bindings to your `PYTHONPATH` enviornment variable. Simply append:
```bash
export PYTHONPATH="${PYTHONPATH}:<path-to-pyidto-py-bindings/bazel-bin>"
```
to your `~/.bashrc` file. This should make `pyidto` available from anywhere.
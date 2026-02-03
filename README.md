# RL Mobile Robotics

Este repositorio contiene un proyecto de navegación autónoma para robótica móvil
basado en Deep Q-Learning (DQN), implementado en ROS 2 y simulado en Stage.

## Requisitos
- Ubuntu 24.04
- ROS 2 Jazzy instalado y configurado
- Python 3.12
- colcon
- Git

## Instalación y configuración del entorno

git clone https://github.com/monchomatu/RL_mobile_robotics.git
cd RL_mobile_robotics
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHON_EXECUTABLE=$(which python)
export PYTHONPATH=$VIRTUAL_ENV/lib/python3.12/site-packages:$PYTHONPATH
source install/setup.bash
colcon build
source install/setup.bash

## Cada vez que se desee iniciar una nueva sesión en terminal se deberá realizar

source .venv/bin/activate
export PYTHON_EXECUTABLE=$(which python)
export PYTHONPATH=$VIRTUAL_ENV/lib/python3.12/site-packages:$PYTHONPATH
source install/setup.bash

## O se puede realizar una modificación al .venv para evitar hacer los exports cada nuevo inicio de sesión:

nano .venv/bin/activate

## Al final del archivo agrega:

export PYTHON_EXECUTABLE=$(which python)
export PYTHONPATH=$VIRTUAL_ENV/lib/python3.12/site-packages:$PYTHONPATH

## Para ejecutar, en la terminal escribir el comando:

ros2 launch stage_ros2 stage.launch.py

## En una segunda terminal: se ejecutará el reset_stage

cd RL_mobile_robotics
source .venv/bin/activate
export PYTHON_EXECUTABLE=$(which python)
export PYTHONPATH=$VIRTUAL_ENV/lib/python3.12/site-packages:$PYTHONPATH
source install/setup.bash
ros2 run rl_stage_env reset_stage.py

## En una tercera terminal: se ejecutará el train node

cd RL_mobile_robotics
source .venv/bin/activate
export PYTHON_EXECUTABLE=$(which python)
export PYTHONPATH=$VIRTUAL_ENV/lib/python3.12/site-packages:$PYTHONPATH
source install/setup.bash
ros2 run rl_stage_env train_node.py



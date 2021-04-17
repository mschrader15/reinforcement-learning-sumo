import os
import sys
import root
from gym.environment import GlobalObservations
from gym.action_space import GlobalTLManager
from gym.simple_controller import SimpleController
import subprocess
from sumolib import checkBinary  # noqa
import traci  # noqa
# import libsumo  # invesigate using this when we are running headless
import runner
import csv

# This is where PATH to SUMO is found
gui = True
if gui:
    sumoBinary = checkBinary('sumo-gui')
else:
    sumoBinary = checkBinary('sumo')


def run(sim_length, sim_step, gym, action_space, controller):
    sim_time = 0
    i = 0

    while sim_time < sim_length:

        counts = gym.update_all_counts()
        action = controller.calc_action(counts)
        # action = action if sim_time > 40 else [2, 6]
        action_space.update_lights(action, sim_time)
        traci.simulationStep()
        sim_time += sim_step

    traci.close()
    sys.stdout.flush()


if __name__ == "__main__":
    # These are the parameters that are passed to SUMO, gotten from the absolute runner file.
    # Probably not a good idea but alas here we are
    SIM_LENGTH = 24 * 3600
    SIM_STEP = 0.5
    TL_IDS = ['63082002', '63082003', '63082004']
    AGG_PERIOD = 600

    SUMO_STRING = ['-n', root.NET_FILE, '-e', str(SIM_LENGTH), '--step-length', str(SIM_STEP),
                   '-r', root.ROUTESAMPLER_ROUTE_FILE, '-a',
                   ", ".join([root.DETECTOR_FILE, root.VEH_DISTRO_FILE, root.CONSTANT_GREEN_TLS_FILE]), ]

    command_line_list = SUMO_STRING

    # start (open) SUMO
    traci.start([sumoBinary] + command_line_list)

    env = GlobalObservations(root.NET_FILE, runner.TL_IDS, name='global', traci_instance=traci)

    action_space = GlobalTLManager(root.INTERSECTION_SETUP, traci)

    sc = SimpleController(runner.TL_IDS, action_space=action_space)

    # run the simulation
    run(runner.SIM_LENGTH, runner.SIM_STEP, env, action_space, sc)

    subprocess.run([os.path.join(os.environ['SUMO_HOME'], 'tools', 'xml', 'xml2csv.py'), 'output.xml'])

from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random


# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    other_robot = env.get_robot((robot_id + 1) % 2)

    # find nearest charging station
    nearest_charge_station = min(env.charge_stations, key=lambda cs:manhattan_distance(robot.position, cs.position))
    cost_to_station = manhattan_distance(nearest_charge_station.position, robot.position)

    if robot.package is not None:
        # if the robot is carrying a package
        package = robot.package
        reward = 2 * manhattan_distance(package.position, package.destination)
        path_cost = manhattan_distance(package.destination, robot.position)
    else:
        # the robot is not carrying a package, find the nearest one
        available_packages = [package for package in env.packages if package.on_board]
        if not available_packages:
            # all the packages have been collected
            return 0
        else:
            nearest_package = min(available_packages, key=lambda package: manhattan_distance(package.position, robot.position))
            reward = 2 * manhattan_distance(nearest_package.position, nearest_package.destination)
            to_package_cost = manhattan_distance(robot.position, nearest_package.position)
            to_dest_cost = manhattan_distance(nearest_package.position, nearest_package.destination)
            path_cost = to_package_cost + to_dest_cost

    # decide - go to charge station or proceed to destination
    if path_cost > robot.battery and robot.battery <= cost_to_station and robot.credit < other_robot.credit:
        path_cost = cost_to_station
        reward = - robot.credit

    trade_off = (robot.credit / (reward+1)) + (robot.battery / (path_cost+1))
    return (reward / (path_cost + 1)) * trade_off


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)
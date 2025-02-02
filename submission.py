from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import time


def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)

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
            nearest_package = min(available_packages, key=lambda pack: manhattan_distance(pack.position, robot.position))
            reward = 2 * manhattan_distance(nearest_package.position, nearest_package.destination)
            to_package_cost = manhattan_distance(robot.position, nearest_package.position)
            to_dest_cost = manhattan_distance(nearest_package.position, nearest_package.destination)
            path_cost = to_package_cost + to_dest_cost

    return robot.credit * 100 + (reward / (path_cost + 1))+robot.battery


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        self.end_time = time.time() + 0.8 * time_limit
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
        d = 2
        step = operators[0]

        while True:
            if time.time() > self.end_time:
                break
            try:
                children_results = [self.RB_minimax(child, 1 - agent_id, d - 1, agent_id) for child in children]
                max_result = max(children_results)
                index_selected = children_results.index(max_result)
                step = operators[index_selected]
                d += 2
            except TimeoutError:
                break

        return step

    def RB_minimax(self, env, robot_id, d, turn):
        if time.time() > self.end_time:
            raise TimeoutError

        if env.done() or d == 0 or env.get_robot(1-turn).battery==0:
            return smart_heuristic(env, robot_id)

        operators = env.get_legal_operators(robot_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(robot_id, op)

        if turn == robot_id:
            cur_max = - float('inf')
            for child in children:
                val = self.RB_minimax(child, 1-robot_id , d-1, turn)
                cur_max = max(val, cur_max)
            return cur_max
        else:
            cur_min = float('inf')
            for child in children:
                val = self.RB_minimax(child, 1-robot_id, d-1, turn)
                cur_min = min(val, cur_min)
            return cur_min

class AgentAlphaBeta(Agent):
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        self.end_time = time.time() + 0.8 * time_limit
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
        d = 2
        step = operators[0]

        while True:
            if time.time() > self.end_time:
                break
            try:
                children_results = [self.RB_AlphaBeta(child, 1 - agent_id, d - 1, agent_id, -float('inf'), float('inf')) for child in children]
                max_result = max(children_results)
                index_selected = children_results.index(max_result)
                step = operators[index_selected]
                d += 2
            except TimeoutError:
                break

        return step

    def RB_AlphaBeta(self, env, robot_id, d, turn, alpha, beta):
        if time.time() > self.end_time:
            raise TimeoutError

        if env.done() or d == 0 or env.get_robot(1 - turn).battery == 0:
            return smart_heuristic(env, robot_id)

        operators = env.get_legal_operators(robot_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(robot_id, op)

        if turn == robot_id:
            cur_max = - float('inf')
            for child in children:
                val = self.RB_AlphaBeta(child, 1 - robot_id, d - 1, turn, alpha, beta)
                cur_max = max(val, cur_max)
                alpha = max(cur_max, alpha)
                if cur_max >= beta:
                    break
            return cur_max
        else:
            cur_min = float('inf')
            for child in children:
                val = self.RB_AlphaBeta(child, 1 - robot_id, d - 1, turn, alpha, beta)
                cur_min = min(val, cur_min)
                beta = min(cur_min, beta)
                if cur_min <= alpha:
                    break
            return cur_min


class AgentExpectimax(Agent):
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        self.end_time = time.time() + 0.8 * time_limit
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
        d = 2
        step = operators[0]

        while True:
            if time.time() > self.end_time:
                break
            try:
                children_results = [self.RB_expectimax(child, 1 - agent_id, d - 1, agent_id) for child in children]
                max_result = max(children_results)
                index_selected = children_results.index(max_result)
                step = operators[index_selected]
                d += 2
            except TimeoutError:
                break

        return step

    def RB_expectimax(self, env, robot_id, d, turn):
        if time.time() > self.end_time:
            raise TimeoutError

        if env.done() or d == 0 or env.get_robot(1 - turn).battery == 0:
            return smart_heuristic(env, robot_id)

        operators = env.get_legal_operators(robot_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(robot_id, op)

        if turn == robot_id:
            cur_max = - float('inf')
            for child in children:
                val = self.RB_expectimax(child, 1 - robot_id, d - 1, turn)
                cur_max = max(val, cur_max)
            return cur_max
        else:
            probabilities = self.calc_prob(operators)
            sum = 0
            for child, prob in zip(children, probabilities):
                sum += prob * self.RB_expectimax(child, 1 - robot_id, d - 1, turn)
            return sum

    def calc_prob(self, operations):
        sum = len(operations)
        if "pick up" in operations:
            sum+=1
        if "move east" in operations:
            sum+=1
        return [
            2 / sum if op == "pick up" or op == "move east" else 1 / sum
            for op in operations
        ]


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
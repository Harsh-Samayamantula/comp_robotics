import argparse
import heapq
import time
import numpy as np
from component_2 import *
from component_3 import *
from component_4 import *
from component_4_1 import *
from component_5 import * 
from component_7 import * 

# Parsing arguments for input configurations
def parse_arguments():
    parser = argparse.ArgumentParser(description="Planner Comparison")
    parser.add_argument('--map', required=True, help="Path to the map file.")
    parser.add_argument('--robot', required=True, choices=['arm', 'car'], help="Type of robot: 'arm' or 'car'.")
    parser.add_argument('--start', nargs='+', type=float, required=True, help="Start configuration.")
    parser.add_argument('--goal', nargs='+', type=float, required=True, help="Goal configuration.")
    return parser.parse_args()

def reconstruct_path(parent, goal):
    path = []
    current = goal

    while current is not None:
        path.append(current)
        current = parent[current]

    path.reverse()  # Reverse the path to get it from start to goal
    return path

def uniform_cost_search2(G, start, goal):
    visited = set()
    queue = []
    #  (cost, node) 
    heapq.heappush(queue, (0, start))

    parent = {start: None}
    costs = {start: 0}

    while queue:
        cost, node = heapq.heappop(queue)
        if node in visited:
            continue

        visited.add(node)
        if node == goal:
            path = reconstruct_path(parent, goal)
            total_cost = costs[goal]
            return path, total_cost

        for neighbor, neighbor_cost in G[node]:
            new_cost = cost + neighbor_cost

            if neighbor not in visited and (neighbor not in costs or new_cost < costs[neighbor]):
                costs[neighbor] = new_cost
                parent[neighbor] = node
                heapq.heappush(queue, (new_cost, neighbor))

    return [], float('inf')

# Main evaluation function
def main_evaluation():
    args = parse_arguments()
    env = scene_from_file(args.map)
    
    if args.robot == "car":
        d = 3
    else:
        d = 2

    k = 6
    max_iter = 500  # Maximum iterations for each planner

    results = {}

    # List of planners to evaluate
    planners = {
        "PRM": lambda: build_prm(args.robot, env, n_samples=max_iter, k=k),
        "RRT": lambda: build_rrt(args.robot, args.start, args.goal, env, max_nodes=max_iter),
        "RRT*": lambda: build_rrt_star(args.robot, args.start, args.goal, env, max_nodes=max_iter)
    }

    # Evaluate each planner directly
    for planner_name, planner_func in planners.items():
        success_count = 0
        path_lengths = []
        times = []

        print(f"Evaluating {planner_name}...")
        
        # Run the planner 10 times
        for _ in range(10):
            start_time = time.time()
            
            # Run the planner to generate graph G
            G = planner_func()
            
            # Find the shortest path in the graph G
            path, total_cost = uniform_cost_search2(G, args.start, args.goal)
            
            end_time = time.time()
            execution_time = end_time - start_time
            times.append(execution_time)

            # Validate and measure the path
            if path:
                path_lengths.append(total_cost)  # Using total_cost from UCS as the path length
                success_count += 1

        # Results summary for the current planner
        success_rate = success_count / 10
        avg_path_length = np.mean(path_lengths) if path_lengths else float('inf')
        avg_time = np.mean(times)

        # Store the results
        results[planner_name] = {
            "Success Rate": success_rate,
            "Average Path Length": avg_path_length,
            "Average Computation Time": avg_time
        }

        print(f"\n{planner_name} Results:")
        print(f"  Success Rate: {success_rate * 100}%")
        print(f"  Average Path Length: {avg_path_length}")
        print(f"  Average Computation Time: {avg_time:.4f} seconds\n")
    
    # Print all results
    for planner_name, metrics in results.items():
        print(f"{planner_name} Final Summary:")
        print(f"  Success Rate: {metrics['Success Rate'] * 100}%")
        print(f"  Average Path Length: {metrics['Average Path Length']}")
        print(f"  Average Computation Time: {metrics['Average Computation Time']:.4f} seconds\n")

if __name__ == '__main__':
    main_evaluation()

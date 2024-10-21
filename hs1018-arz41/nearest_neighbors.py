import numpy as np
import argparse

def distance(config1, config2, robot_type):
    if robot_type == "arm":
        return np.linalg.norm(np.array(config1) - np.array(config2))
    elif robot_type == "freeBody":
        pos_dist = np.linalg.norm(np.array(config1[:2]) - np.array(config2[:2]))
        angle_dist = min(abs(config1[2] - config2[2]), 2*np.pi - abs(config1[2] - config2[2]))
        return pos_dist + angle_dist

def nearest_neighbors(robot_type, target, k, configs):
    distances = [distance(target, config, robot_type) for config in configs]
    return np.argsort(distances)[:k]

def load_configs(filename):
    with open(filename, 'r') as f:
        return [list(map(float, line.strip().split())) for line in f]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nearest Neighbors Search")
    parser.add_argument("--robot", type=str, choices=["arm", "freeBody"], required=True, help="Robot type")
    parser.add_argument("--target", type=float, nargs='+', required=True, help="Target configuration")
    parser.add_argument("-k", type=int, required=True, help="Number of nearest neighbors")
    parser.add_argument("--configs", type=str, required=True, help="File containing configurations")
    
    args = parser.parse_args()
    
    configs = load_configs(args.configs)
    nearest = nearest_neighbors(args.robot, args.target, args.k, configs)
    
    print(f"Target configuration: {args.target}")
    print(f"{args.k} nearest neighbors:")
    for i, idx in enumerate(nearest):
        print(f"{i+1}. Configuration: {configs[idx]}, Distance: {distance(args.target, configs[idx], args.robot)}")
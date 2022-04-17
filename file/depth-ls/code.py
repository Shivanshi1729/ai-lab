def DLS(start, goal, path, level, maxD):
    print('Current level: ', level)
    path.append(start)
    if start == goal:
        return path
    # max depth reached
    if level == maxD:
        return False
    # visit its children
    for child in graph[start]:
        # recursively find a sol
        if DLS(child, goal, path, level+1, maxD):
            return path
        path.pop()
    return False


graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F', 'G'],
    'D': ['H', 'I'],
    'E': ['J', 'K'],
    'F': ['L', 'M'],
    'G': ['N', 'O'],
    'H': [],
    'I': [],
    'J': [],
    'K': [],
    'L': [],
    'M': [],
    'N': [],
    'O': []
}
start = 'A'
goal = 'D'
maxD = 4
path = []
res = DLS(start, goal, path, 0, maxD)
if(res):
    print("Path to goal node available")
    print("Path", path)
else:
    print("No path available for the goal node in given depth limit")

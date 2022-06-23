import math
from queue import PriorityQueue

def branch_and_bound(graph, start, goal):
    bound = math.inf
    finalPath = None

    pq = PriorityQueue()
    pq.put([0, start, set(), [start]])

    while(not pq.empty()):
        current = pq.get()

        if current[0]>=bound:
            continue

        length = current[0]
        node = current[1]
        path = current[3]

        vis = set(list(current[2]))
        vis.add(node)

        if node==goal:
            if length<bound:
                bound = length
                finalPath = path.copy()
        else:
            for neighbour, weight in graph[node].items():
                if neighbour not in current[2]:
                    npath = path.copy()
                    npath.append(neighbour)
                    pq.put([length+weight, neighbour, vis, npath])

    return bound, finalPath

graph = {
    'S': {'A':4, 'B':5},
    'A': {'B':3, 'D':8, 'E':15},
    'B': {'A':3, 'C':4, 'D':6},
    'C': {'B':4, 'D':4, 'G':7},
    'D': {'B':6, 'A':8, 'C':4, 'G':2, 'E':2},
    'E': {'A':15, 'D':2},
    'G': {'D':2, 'C':7} 
}

bound, path = branch_and_bound(graph, 'S', 'G')

print(bound)
print(path)

def DFS(graph, start, goal, visited):
    visited.add(start)
    print(start, end=" ")
    if start == goal:
        return
    for neighbor in graph[start]:
        if neighbor not in visited:
            DFS(graph, neighbor, goal, visited)


graph = {
    'S': {'A': 3, 'B': 2},
    'A': {'C': 4, 'D': 1, 'S': 3},
    'B': {'E': 3, 'F': 1, 'S': 2},
    'C': {'A': 4},
    'D': {'A': 1},
    'E': {'B': 3, 'H': 5},
    'F': {'B': 1, 'I': 2, 'G': 3},
    'G': {'F': 3},
    'I': {'F': 2},
    'H': {'E': 5}
}

DFS(graph, 'S', 'G', set())

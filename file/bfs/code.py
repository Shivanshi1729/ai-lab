from queue import Queue


def BFS(graph, source, destination, visited):
    q = Queue()
    q.put(source)
    visited.add(source)
    while not q.empty():
        current = q.get()
        print(current, end=" ")
        if(current == destination):
            return
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                q.put(neighbor)


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

BFS(graph, 'S', 'G', set())

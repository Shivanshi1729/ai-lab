from queue import PriorityQueue


def Beam_Search(grid, heuristic, start, end, beamWidth):
    visited = set([start])
    pq = PriorityQueue()
    pq.put([heuristic[start], start])

    while(not pq.empty()):
        current = pq.get()

        print(current[1], end=" ")

        if current[1] == end:
            break

        npq = pq
        pq = PriorityQueue()

        for neighbour in grid[current[1]]:
            if neighbour not in visited:
                pq.put([heuristic[neighbour], neighbour])
                visited.add(neighbour)

        for i in range(beamWidth):
            if not npq.empty():
                pq.put(npq.get())


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
    'H': {'E': 5},
}

heuristic = {
    'S': 13,
    'A': 12,
    'B': 4,
    'C': 7,
    'D': 3,
    'E': 8,
    'F': 2,
    'G': 0,
    'H': 4,
    'I': 9
}

source = 'S'
destination = 'G'
beamWidth = 2

Beam_Search(graph, heuristic, source, destination, beamWidth)

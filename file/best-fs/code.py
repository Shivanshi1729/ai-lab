from queue import PriorityQueue

# Filling adjacency matrix with empty arrays
vertices = 14
graph = [[] for i in range(vertices)]


def add_edge(x, y, cost):
    graph[x].append((y, cost))
    graph[y].append((x, cost))


def best_first_search(source, target, vertices):
    visited = [0] * vertices
    pq = PriorityQueue()
    pq.put((0, source))
    visited[source] = True
    print("Path: ")
    while not pq.empty():
        u = pq.get()[1]
        # Displaying the path having the lowest cost
        print(u, end=" ")
        if u == target:
            break

        for v, c in graph[u]:
            if not visited[v]:
                visited[v] = True
                pq.put((c, v))
    print()


if __name__ == '__main__':
    add_edge(0, 1, 1)
    add_edge(0, 2, 8)
    add_edge(1, 2, 12)
    add_edge(1, 4, 13)
    add_edge(2, 3, 6)
    add_edge(4, 3, 3)

    source = 0
    target = 2
    best_first_search(source, target, vertices)

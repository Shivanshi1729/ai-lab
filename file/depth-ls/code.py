def depth_limited_search(graph, start, end, visited, level, limit):
    if level == limit:
        return
    visited.add(start)
    print(start, end=" ")
    if start == end:
        return
    for neighbor in graph[start]:
        if neighbor not in visited:
            depth_limited_search(graph, neighbor, end,
                                 visited, level + 1, limit)


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

depth_limited_search(graph, 'S', 'G', set(), 0, 3)

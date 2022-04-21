def AStar(graph, heuristic, source, destination):
    open_list = set([source])
    close_list = set([])

    g = {
        source: 0
    }

    parent = {
        source: source
    }

    while len(open_list) > 0:
        n = None

        for v in open_list:
            if n == None or g[n]+heuristic[n] > g[v]+heuristic[v]:
                n = v

        if n == destination:
            reconstructin = []

            while parent[n] != n:
                reconstructin.append(n)
                n = parent[n]

            reconstructin.append(source)
            reconstructin.reverse()

            print("Path Found: ", reconstructin)

            return reconstructin

        for m, weight in graph[n].items():

            if m not in open_list and m not in close_list:
                open_list.add(m)
                g[m] = g[n] + weight
                parent[m] = n
            else:
                if g[m] > g[n]+weight:
                    g[m] = g[n] + weight
                    parent[m] = n

                    if m in close_list:
                        close_list.remove(m)
                        open_list.add(m)

        open_list.remove(n)
        close_list.add(n)

    print('Path does not exist')
    return None


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

AStar(graph, heuristic, source, destination)

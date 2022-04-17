def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)

    print(start)

    for next in graph[start]:
        if(next not in visited):
            dfs(graph, next, visited)
    return visited


graph = {'0': set(['1', '2']),
         '1': set(['0', '3', '5']),
         '2': set(['0', '4']),
         '3': set(['1']),
         '4': set(['2']),
         '5': set(['1']),
         }

dfs(graph, '0')

library(igraph)

g <- graph.formula(A-B,A-C,B-C,B-D,C-E, D-E)
betweenness(g)

edge.betweenness(g)
tkplot(g, vertex.color="red") #plot(g)
lg <- tk_coords(1)

#V(g) #vertices
#E(g) #edges
get.adjacency(g)

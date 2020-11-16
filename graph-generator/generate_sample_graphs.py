import os
import networkx as nx
from random import randint
from numpy.random import uniform

CURRENT_DIRECTORY_OF_SCRIPT = os.path.dirname(os.path.realpath(__file__))
INITIAL_OUTPUT_PATH = os.path.join(CURRENT_DIRECTORY_OF_SCRIPT, "created_graphs_output")

def cellular_network(cluster_size, clusters, p, seed, directed):
    def mapping(x):
        return x + cluster_size

    cellular_network_graph = \
        nx.gnp_random_graph(n = cluster_size, p = p, seed = seed, directed = directed)

    for cluster_ix in range(1, clusters):
        F = nx.relabel_nodes(cellular_network_graph, mapping)
        T = nx.compose(F, cellular_network_graph)
        cellular_network_graph = T

    for cluster_ix_outer in range(0, clusters):
        for cluster_ix_inner in range(0, clusters):
            if (cluster_ix_outer >= cluster_ix_inner):
                outer_node_selected = \
                    randint(cluster_ix_outer * cluster_size, 9 + cluster_ix_outer * cluster_size)
                inner_node_selected = \
                    randint(cluster_ix_inner * cluster_size, 9 + cluster_ix_inner * cluster_size)
                cellular_network_graph.add_edge(outer_node_selected, inner_node_selected)

    return(cellular_network_graph)

# CONFIGURATION SETTINGS
# Note: For config help, check out https://networkx.github.io/documentation/networkx-2.0/reference/generators.html

N=2   #Number of graphs required
n=20   #Number of nodes in the graph
graph_type="cellular"
specify_p=True
specify_k=False
p_value=0.2
k_value=5
count=0

if(graph_type=="scale_free"):
        for i in range(N):
            count+=1
            G=nx.barabasi_albert_graph(n=n , m = 5, seed = randint(1, 100))      # m should be <= n
            final_output_path = os.path.join(INITIAL_OUTPUT_PATH, "graph-" \
																+ graph_type \
																+ "_nodes-" \
																+ str(n) \
																+ "sample-" \
																+ str(count) \
																+ ".edgelist")
            nx.write_edgelist(G, final_output_path)

elif(graph_type=="small_world"):
    if(specify_p==True):
        if(specify_k==True):
            for i in range(N):
                count+=1
                k= k_value
                p = p_value
                G=nx.watts_strogatz_graph(n=n, k= k, p = p, seed = randint(1, 100))      # 1<=k<=n-1; 0<=p<1
                final_output_path = os.path.join(INITIAL_OUTPUT_PATH, "graph-" \
																	+ graph_type \
																	+ "_nodes-" \
																	+ str(n) \
																	+ "_k-" \
																	+ str(k) \
																	+ "_p-" \
																	+ str(p) \
																	+"_sample-" \
																	+ str(count) \
																	+ ".edgelist")
                nx.write_edgelist(G, final_output_path)
        else:
            for i in range(N):
                count+=1
                k = randint(1, n-1)
                p=p_value
                seed = randint(1, 100)
                G=nx.watts_strogatz_graph(n=n, k = k, p= p, seed = randint(1, 100))      # 1<=k<=n-1; 0<=p<1
                final_output_path = os.path.join(INITIAL_OUTPUT_PATH, "graph-" \
																	+ graph_type \
																	+ "_nodes-" \
																	+ str(n) \
																	+ "_k-" \
																	+ str(k) \
																	+ "_p-" \
																	+ str(p) \
																	+"_sample-" \
																	+ str(count) \
																	+ ".edgelist")
                nx.write_edgelist(G, final_output_path)
    else:
        if(specify_k==True):
            for i in range(N):
                count+=1
                k=k_value
                p = uniform(0,1)
                seed = randint(1, 100)
                G=nx.watts_strogatz_graph(n=n , k=k, p = p, seed = randint(1, 100))      # 1<=k<=n-1; 0<=p<1
                final_output_path = os.path.join(INITIAL_OUTPUT_PATH, "graph-" \
																	+ graph_type \
																	+ "_nodes-" \
																	+ str(n) \
																	+ "_k-" \
																	+ str(k) \
																	+ "_p-" \
																	+ str(p) \
																	+"_sample-" \
																	+ str(count) \
																	+ ".edgelist")
                nx.write_edgelist(G, final_output_path)
        else :
            for i in range(N):
                count+=1
                k = randint(1, n-1)
                p = uniform (0,1)
                seed = randint(1, 100)
                G=nx.watts_strogatz_graph(n, k = k, p = p, seed = randint(1, 100))      # 1<=k<=n-1; 0<=p<1
                final_output_path = os.path.join(INITIAL_OUTPUT_PATH, "graph-" \
																	+ graph_type \
																	+ "_nodes-" \
																	+ str(n) \
																	+ "_k-" \
																	+ str(k) \
																	+ "_p-" \
																	+ str(p) \
																	+"_sample-" \
																	+ str(count) \
																	+ ".edgelist")
                nx.write_edgelist(G, final_output_path)

elif(graph_type=="random"):
    if(specify_p==True):
        for i in range(N):
            count+=1
            p=p_value
            G=nx.gnp_random_graph(n=n , p=p, seed = randint(1, 100), directed=False)
            final_output_path = os.path.join(INITIAL_OUTPUT_PATH, "graph-" \
																+ graph_type \
																+ "_nodes-" \
																+ str(n) \
																+ "_p-" \
																+ str(p) \
																+"_sample-" \
																+ str(count) \
																+ ".edgelist")
            nx.write_edgelist(G, final_output_path)
    else:
        for i in range(N):
            count+=1
            p = uniform (0,1)
            seed = randint(1, 100)
            G=nx.gnp_random_graph(n=n, p = p, seed = randint(1, 100), directed=False)
            final_output_path = os.path.join(INITIAL_OUTPUT_PATH, "graph-" \
																+ graph_type \
																+ "_nodes-" \
																+ str(n) \
																+ "_p-" \
																+ str(p) \
																+"_sample-" \
																+ str(count) \
																+ ".edgelist")
            nx.write_edgelist(G, final_output_path)

elif(graph_type=="cellular"):
    if(specify_p==True):
        for i in range(N):
            count+=1
            p = p_value
            G=cellular_network(cluster_size = 10, clusters = 5, p = p, seed = randint(0, 100), directed = False)
            final_output_path = os.path.join(INITIAL_OUTPUT_PATH, "graph-" \
																+ graph_type \
																+ "_p-" \
																+ str(p) \
																+"_sample-" \
																+ str(count) \
																+ ".edgelist")
            nx.write_edgelist(G, final_output_path)
    else:
        for i in range(N):
            count+=1
            p = uniform (0,1)
            seed = randint(0, 100)
            G=cellular_network(cluster_size = 10, clusters = 5, p = p, seed = randint(0, 100), directed = False)
            final_output_path = os.path.join(INITIAL_OUTPUT_PATH, "graph-" \
																+ graph_type \
																+ "_p-" \
																+ str(p) \
																+"_sample-" \
																+ str(count) \
																+ ".edgelist")
            nx.write_edgelist(G, final_output_path)

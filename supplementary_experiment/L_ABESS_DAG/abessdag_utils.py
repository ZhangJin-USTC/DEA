import numpy as np
import networkx as nx
import pandas as pd
import random
import math
import abess

def graphgen(p, q, min_w, max_w, graph_type="ER"):
    if (graph_type=="ER"):
        graphun=nx.erdos_renyi_graph(p, 2*q/(p-1))
    if (graph_type=="BA"):
        graphun=nx.barabasi_albert_graph(p, q)
    names = ["X"+ str(i) for i in range(p)]
    graph = nx.DiGraph()
    graph.add_nodes_from(names)
    for e in graphun.edges():
        w = random.uniform(min_w,max_w) * (1 if random.random() < 0.5 else -1)
        graph.add_weighted_edges_from([("X"+str(e[0]),"X"+str(e[1]),w)])
    return graph

def simdag(n, graph, sd, error_type="Gaussian"):
    data = pd.DataFrame(columns=list(nx.topological_sort(graph)))
    for i in list(nx.topological_sort(graph)):
        data[i] = np.random.normal(0, scale=sd, size=n)
        if (error_type == "Uniform"):
            data[i] = np.random.uniform(-math.sqrt(3) * sd,
                                        math.sqrt(3) * sd,
                                        size=n)
        if graph.in_degree(i) != 0:
            for suc in [j for j in list(graph) if i in list(graph[j])]:
                data[i] += (graph[suc][i]["weight"] * data[suc])
    data = data.sample(frac=1, axis=1)  # Shuffle the columns
    return (data)

def stdupdate(i,ancestors,data):
    model = abess.LinearRegression()
    model.fit(ancestors.values, data[i])
    ind = np.nonzero(model.coef_)
    res = data[i] - model.predict(ancestors.values)
    return ([i, (res**2).sum(), ind])

def edge_summary (graph, est_edge):
    real_edge_len = len(graph.edges)
    esti_edge_len = est_edge.shape[0]
    realedges = pd.DataFrame(data=None, columns=["From","To"])
    for i in graph.edges:
        realedges.loc[len(realedges)]=[i[0],i[1]]
    df = pd.merge(est_edge, realedges, on=['From','To'], how='outer', indicator='Exist')
    True_edge_number = sum(df['Exist']=='both')
    flip_realedges = realedges.rename({'From': 'To', 'To': 'From'}, axis=1)
    fldf = pd.merge(est_edge, flip_realedges, on=['From','To'], how='outer', indicator='Exist')
    Flip_edge_number = sum(fldf['Exist']=='both')
    print("The number of edges in true graph is ", real_edge_len)
    print("We discovered ", esti_edge_len, " edges with our abess-dag algorithm")
    print("Recall (the percentage of true edges discovered): ", round(True_edge_number/real_edge_len*100,2), "%, which means that ",True_edge_number, " edges were perfectly found.")
    print(Flip_edge_number," edge(s) was detected with wrong direction.")
    print("FDR (the percentage of estimated edges that are either flipped or not present in the true graph): ", round((1-True_edge_number/esti_edge_len)*100,2),"%")

def edge_stat (ginfo, est_edge):
    real_edge_len = ginfo.shape[0]
    esti_edge_len = est_edge.shape[0]
    realedges = ginfo[['From','To']]
    df = pd.merge(est_edge, realedges, on=['From','To'], how='outer', indicator='Exist')
    True_edge_number = sum(df['Exist']=='both')
    flip_realedges = realedges.rename({'From': 'To', 'To': 'From'}, axis=1)
    fldf = pd.merge(est_edge, flip_realedges, on=['From','To'], how='outer', indicator='Exist')
    Flip_edge_number = sum(fldf['Exist']=='both')
    TPR=round(True_edge_number/real_edge_len*100,2)
    FDR=round((1-True_edge_number/esti_edge_len)*100,2)
    SHD=(esti_edge_len-True_edge_number-Flip_edge_number) + (real_edge_len-True_edge_number)
    return TPR,FDR,SHD,Flip_edge_number

def batchimport(kset,causes,effects):
    for i in causes:
        for j in effects:
            kset.loc[len(kset)] = [i,j]
    return kset

def iscompatible(kset):
    kset_tuple = kset.apply(tuple,axis=1).tolist()
    G=nx.DiGraph()
    G.add_edges_from(kset_tuple)
    result = nx.is_directed_acyclic_graph(G)
    return result

def mostcauses(kset):
    ksetv = kset.Cause.value_counts()
    mostc = ksetv[ksetv==ksetv.max()].index
    return mostc

def pps(i,kset,edges):
    edges = pd.DataFrame(edges, columns=["From", "To"])
    kset_subset = kset.loc[kset['Cause']==i]
    props1 = set(kset_subset.Effect)
    props = props1
    edges_tuple = edges.apply(tuple,axis=1).tolist()
    G=nx.DiGraph()
    G.add_edges_from(edges_tuple)
    for j in props1:
        if G.has_node(j):
            props = props.union(nx.descendants(G,j)) 
    return props

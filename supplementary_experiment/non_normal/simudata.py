import math
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import gumbel_r


def simdag(n, graph, sd, error_type="Gaussian"):
    data = pd.DataFrame(columns=list(nx.topological_sort(graph)))
    for i in list(nx.topological_sort(graph)):
        data[i] = np.random.normal(0, scale=sd, size=n)
        if (error_type == "Uniform"):
            data[i] = np.random.uniform(-math.sqrt(3) * sd,
                                        math.sqrt(3) * sd,
                                        size=n)
        if (error_type == "Gumbel"):
            data[i] = gumbel_r.rvs(loc=-sd*math.sqrt(6)*0.577215665/math.pi, 
                                   scale=sd*math.sqrt(6)/math.pi, size=n)
        if (error_type == "Laplace"):
            data[i] = np.random.laplace(0, sd/math.sqrt(2), size=n)
        if (error_type == "t"):
            data[i] = sd* np.random.standard_t(df=3, size=n) / math.sqrt(3)
        if graph.in_degree(i) != 0:
            for suc in [j for j in list(graph) if i in list(graph[j])]:
                data[i] += (graph[suc][i]["weight"] * data[suc])
    data = data.sample(frac=1, axis=1)  # Shuffle the columns
    return (data)

for k in range(100):
    pth_info = os.getcwd()+r"\data\ER2\graph\ER2_p30_{k}.csv".format(k=k)
    ginfo = pd.read_csv(pth_info)
    names = ["X"+ str(i) for i in range(30)]
    graph = nx.DiGraph()
    graph.add_nodes_from(names)
    for i in range(len(ginfo)):
        graph.add_weighted_edges_from([(ginfo.iloc[i,1],ginfo.iloc[i,2],ginfo.iloc[i,3])])
    pth_data = os.getcwd()+"\data\ER2"

    data = simdag(30, graph, sd=0.3, error_type="Uniform")
    data.to_csv(pth_data+r"\Uniform\n30\ER2_Uniform_p30_n30_{k}.csv".format(k=k))
    data = simdag(100, graph, sd=0.3)
    data.to_csv(pth_data+r"\Uniform\n100\ER2_Uniform_p30_n100_{k}.csv".format(k=k))
    data = simdag(300, graph, sd=0.3)
    data.to_csv(pth_data+r"\Uniform\n300\ER2_Uniform_p30_n300_{k}.csv".format(k=k))

    data = simdag(30, graph, sd=0.3, error_type="Gumbel")
    data.to_csv(pth_data+r"\Gumbel\n30\ER2_Gumbel_p30_n30_{k}.csv".format(k=k))
    data = simdag(100, graph, sd=0.3)
    data.to_csv(pth_data+r"\Gumbel\n100\ER2_Gumbel_p30_n100_{k}.csv".format(k=k))
    data = simdag(300, graph, sd=0.3)
    data.to_csv(pth_data+r"\Gumbel\n300\ER2_Gumbel_p30_n300_{k}.csv".format(k=k))

    data = simdag(30, graph, sd=0.3, error_type="Laplace")
    data.to_csv(pth_data+r"\Laplace\n30\ER2_Laplace_p30_n30_{k}.csv".format(k=k))
    data = simdag(100, graph, sd=0.3)
    data.to_csv(pth_data+r"\Laplace\n100\ER2_Laplace_p30_n100_{k}.csv".format(k=k))
    data = simdag(300, graph, sd=0.3)
    data.to_csv(pth_data+r"\Laplace\n300\ER2_Laplace_p30_n300_{k}.csv".format(k=k))

    data = simdag(30, graph, sd=0.3, error_type="t")
    data.to_csv(pth_data+r"\t\n30\ER2_t_p30_n30_{k}.csv".format(k=k))
    data = simdag(100, graph, sd=0.3)
    data.to_csv(pth_data+r"\t\n100\ER2_t_p30_n100_{k}.csv".format(k=k))
    data = simdag(300, graph, sd=0.3)
    data.to_csv(pth_data+r"\t\n300\ER2_t_p30_n300_{k}.csv".format(k=k))




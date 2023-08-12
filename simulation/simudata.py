p=300

# ER2
for k in range(100):
    g=graphgen(p, 2, 0.5, 1, graph_type="ER")
    edges = g.edges(data=True)
    edges = [(i[0], i[1], i[2]["weight"]) for i in edges]
    aa = pd.DataFrame(edges, columns=["From", "To", "Weight"])
    aa.to_csv(os.getcwd()+r"\data\ER2\graph\ER2_p300_{k}.csv".format(k=k))
    data = simdag(100, g, sd=0.3)
    data.to_csv(os.getcwd()+r"\data\ER2\p300\n100\ER2_p300_n100_{k}.csv".format(k=k))
    data2 = simdag(300, g, sd=0.3)
    data2.to_csv(os.getcwd()+r"\data\ER2\p300\n300\ER2_p300_n300_{k}.csv".format(k=k))
    data3 = simdag(1000, g, sd=0.3)
    data3.to_csv(os.getcwd()+r"\data\ER2\p300\n1000\ER2_p300_n1000_{k}.csv".format(k=k))

# ER5
for k in range(100):
    g=graphgen(p, 5, 0.5, 1, graph_type="ER")
    edges = g.edges(data=True)
    edges = [(i[0], i[1], i[2]["weight"]) for i in edges]
    aa = pd.DataFrame(edges, columns=["From", "To", "Weight"])
    aa.to_csv(os.getcwd()+r"\data\ER5\graph\ER5_p300_{k}.csv".format(k=k))
    data = simdag(100, g, sd=0.3)
    data.to_csv(os.getcwd()+r"\data\ER5\p300\n100\ER5_p300_n100_{k}.csv".format(k=k))
    data2 = simdag(300, g, sd=0.3)
    data2.to_csv(os.getcwd()+r"\data\ER5\p300\n300\ER5_p300_n300_{k}.csv".format(k=k))
    data3 = simdag(1000, g, sd=0.3)
    data3.to_csv(os.getcwd()+r"\data\ER5\p300\n1000\ER5_p300_n1000_{k}.csv".format(k=k))

# BA2
for k in range(100):
    g=graphgen(p, 2, 0.5, 1, graph_type="BA")
    edges = g.edges(data=True)
    edges = [(i[0], i[1], i[2]["weight"]) for i in edges]
    aa = pd.DataFrame(edges, columns=["From", "To", "Weight"])
    aa.to_csv(os.getcwd()+r"\data\BA2\graph\BA2_p300_{k}.csv".format(k=k))
    data = simdag(100, g, sd=0.3)
    data.to_csv(os.getcwd()+r"\data\BA2\p300\n100\BA2_p300_n100_{k}.csv".format(k=k))
    data2 = simdag(300, g, sd=0.3)
    data2.to_csv(os.getcwd()+r"\data\BA2\p300\n300\BA2_p300_n300_{k}.csv".format(k=k))
    data3 = simdag(1000, g, sd=0.3)
    data3.to_csv(os.getcwd()+r"\data\BA2\p300\n1000\BA2_p300_n1000_{k}.csv".format(k=k))

# BA5
for k in range(100):
    g=graphgen(p, 5, 0.5, 1, graph_type="BA")
    edges = g.edges(data=True)
    edges = [(i[0], i[1], i[2]["weight"]) for i in edges]
    aa = pd.DataFrame(edges, columns=["From", "To", "Weight"])
    aa.to_csv(os.getcwd()+r"\data\BA5\graph\BA5_p300_{k}.csv".format(k=k))
    data = simdag(100, g, sd=0.3)
    data.to_csv(os.getcwd()+r"\data\BA5\p300\n100\BA5_p300_n100_{k}.csv".format(k=k))
    data2 = simdag(300, g, sd=0.3)
    data2.to_csv(os.getcwd()+r"\data\BA5\p300\n300\BA5_p300_n300_{k}.csv".format(k=k))
    data3 = simdag(1000, g, sd=0.3)
    data3.to_csv(os.getcwd()+r"\data\BA5\p300\n1000\BA5_p300_n1000_{k}.csv".format(k=k))
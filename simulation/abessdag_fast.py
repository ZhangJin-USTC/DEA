import pandas as pd
import abess
import numpy as np
import joblib
from abessdag_utils import iscompatible,mostcauses,pps

def fit_model(args):
    model, X, y = args
    model.fit(X, y)
    return model

def abessdag(data_in, nu=0, kset=pd.DataFrame(columns=["Cause", "Effect"])
                   , wkset=pd.DataFrame(columns=["Cause", "Effect"]), strategy="DB"):
    data = data_in.copy()
    if (kset.empty) and (wkset.empty):
        if (nu==0):
            data = data_in.copy()
            order = []
            ancestors = data[[data.std().idxmin()]]
            order.append(data.std().idxmin())
            data = data.drop(data.std().idxmin(), axis=1)
            edges = []
            p = data.shape[1]
            for _ in range(p):
                stdlist = []
                ancestors_values = ancestors.values
                for i in data.columns:
                    model = abess.LinearRegression()
                    model.fit(ancestors_values, data[i])
                    ind = np.nonzero(model.coef_)[0]
                    if ancestors.shape[1]==1:
                        res = data[i] - model.predict(ancestors.values)[:,1]
                    else:
                        res = data[i] - model.predict(ancestors_values)   
                    stdlist.append((i, res.std(), ind))
                mini, _, minind = min(stdlist, key=lambda x: x[1])
                order.append(mini)
                for j in minind:
                    edges.append((ancestors.columns[j], mini))
                ancestors = pd.concat((ancestors, data[mini]), axis=1)
                data = data.drop(mini, axis=1)
            edges = pd.DataFrame(edges, columns=["From", "To"])
            return order, edges
        if (nu<0):
            raise ValueError("The threshold can't be negative.")
        if (nu>0):
            n = data.shape[0]
            order = []
            w=(data**2).sum()
            choose_ind = w[w<=w.min()+nu*n].index
            ancestors = data[choose_ind]
            data = data.drop(choose_ind,axis=1)
            order.append(choose_ind.tolist())
            edges = []
    
            if (len(choose_ind)==1):
                stdlist = pd.DataFrame(data=None,columns=['Vertex','loss','ind'])
                models = []
                for i in data.columns:
                    model = abess.LinearRegression()
                    models.append(model)
                results = joblib.Parallel(n_jobs=-1)(
                    joblib.delayed(fit_model)(args)
                    for args in zip(models, [ancestors.values]*len(models), [data[i] for i in data.columns])
                )
                for i, result in enumerate(results):
                    ind = np.nonzero(result.coef_)
                    res = data[data.columns[i]] - result.predict(ancestors.values)[:,1]
                    stdlist.loc[len(stdlist)] = [data.columns[i], (res**2).sum(), ind]
                w = stdlist["loss"]
                choose_ind = w[w<=w.min()+nu*n].index
                vertex_ind = stdlist["Vertex"][choose_ind].tolist()
                order.append(vertex_ind)
                for k in vertex_ind:
                    minind = stdlist.loc[stdlist["Vertex"]==k,"ind"].item()[0]
                    for j in minind:
                        edges.append((ancestors.columns[j], k))
                    ancestors.insert(loc=0 ,column=k, value=data[k])
                    data = data.drop(k, axis=1)
      
            while (data.shape[1]>0):
                stdlist = pd.DataFrame(data=None,columns=['Vertex','loss','ind'])
                models = []
                for i in data.columns:
                    model = abess.LinearRegression()
                    models.append(model)
                results = joblib.Parallel(n_jobs=-1)(
                    joblib.delayed(fit_model)(args)
                    for args in zip(models, [ancestors.values]*len(models), [data[i] for i in data.columns])
                )
                for i, result in enumerate(results):
                    ind = np.nonzero(result.coef_)
                    res = data[data.columns[i]] - result.predict(ancestors.values)
                    stdlist.loc[len(stdlist)] = [data.columns[i], (res**2).sum(), ind]
                w = stdlist["loss"]
                choose_ind = w[w<=w.min()+nu*n].index
                vertex_ind = stdlist["Vertex"][choose_ind].tolist()
                order.append(vertex_ind)
                for k in vertex_ind:
                    minind = stdlist.loc[stdlist["Vertex"]==k,"ind"].item()[0]
                    for j in minind:
                        edges.append((ancestors.columns[j], k))
                    ancestors = pd.concat((ancestors,data[k]),axis=1)
                    data = data.drop(k, axis=1)
            edges = pd.DataFrame(edges, columns=["From", "To"])
            return order, edges
    
    else:
        order = []
        for col in ['Cause', 'Effect']:
            if not set(getattr(kset, col)).issubset(set(data.columns)) or not set(getattr(wkset, col)).issubset(set(data.columns)):
                raise ValueError("Knowledge set should contain the names of variables.")
        if (not iscompatible(kset)) and (not (kset.empty)):
            raise ValueError("Strong knowledge set must be compatible.")
        if (strategy == "KB") and (kset.empty):
            data_frac = data[mostcauses(wkset)]
            ancestors = data_frac[[data_frac.std().idxmin()]]
            order.append(data_frac.std().idxmin())
            wkset = wkset.drop(wkset[wkset['Cause'] == data_frac.std().idxmin()].index)
            data = data.drop(data_frac.std().idxmin(), axis=1)
            edges = []
            p = data.shape[1]
            for _ in range(p):
                stdlist = []
                desc = data.columns if (wkset.empty) else mostcauses(wkset)
                for i in desc:
                    props = pps(i,wkset,edges)
                    props = props.intersection(ancestors.columns)  # Prohibited parent set
                    model = abess.LinearRegression()
                    anc = ancestors.drop(columns=props)
                    if anc.empty:
                        stdlist.append((i, data[i].std(), np.empty((0,), dtype=np.int64)))
                    if anc.shape[1]==1:
                        model.fit(anc.values,data[i])  # Use some of the ancestors to do variable selection
                        ind = np.nonzero(model.coef_)[0]
                        res = data[i] - model.predict(anc.values)[:,1]
                    if anc.shape[1]>=2:
                        model.fit(anc.values,data[i])  # Use some of the ancestors to do variable selection
                        ind = np.nonzero(model.coef_)[0]
                        res = data[i] - model.predict(anc.values)
                    stdlist.append((i, res.std(), ind))
                mini, _, minind = min(stdlist, key=lambda x: x[1])
                order.append(mini)
                props = pps(mini,wkset,edges)
                props = props.intersection(ancestors.columns)
                anc = ancestors.drop(columns=props)
                for j in minind:
                    edges.append((anc.columns[j],mini))
                ancestors = pd.concat((ancestors,data[mini]),axis=1)
                data = data.drop(mini, axis=1)
                wkset = wkset.drop(wkset[wkset['Cause'] == mini].index)
            return order, edges
        else:    
            data_frac = data[data.columns.drop(set(kset.Effect))]
            ancestors = data_frac[[data_frac.std().idxmin()]]
            order.append(data_frac.std().idxmin())
            kset = kset.drop(kset[kset['Cause'] == data_frac.std().idxmin()].index)
            data = data.drop(data_frac.std().idxmin(), axis=1)
            edges = []
            p = data.shape[1]
            for _ in range(p):
                stdlist = []
                cols_to_drop = set(kset.Effect) & set(data.columns)
                for i in data.columns.drop(cols_to_drop):
                    props = pps(i,wkset,pd.DataFrame(edges))
                    props = props.intersection(ancestors.columns)  # Prohibited parent set
                    model = abess.LinearRegression()
                    anc = ancestors.drop(columns=props)
                    if anc.empty:
                        stdlist.append((i, data[i].std(), np.empty((0,), dtype=np.int64)))
                    if anc.shape[1]==1:
                        model.fit(anc.values,data[i])  # Use some of the ancestors to do variable selection
                        ind = np.nonzero(model.coef_)[0]
                        res = data[i] - model.predict(anc.values)[:,1]
                    if anc.shape[1]>=2:
                        model.fit(anc.values,data[i])  # Use some of the ancestors to do variable selection
                        ind = np.nonzero(model.coef_)[0]
                        res = data[i] - model.predict(anc.values)
                    stdlist.append((i, res.std(), ind))
                mini, _, minind = min(stdlist, key=lambda x: x[1])
                order.append(mini)
                props = pps(mini,wkset,pd.DataFrame(edges))
                props = props.intersection(ancestors.columns)
                anc = ancestors.drop(columns=props)
                for j in minind:
                    edges.append((anc.columns[j],mini))
                ancestors = pd.concat((ancestors,data[mini]),axis=1)
                data = data.drop(mini, axis=1)
                kset = kset.drop(kset[kset['Cause'] == mini].index)
            
            edges = pd.DataFrame(edges, columns=['From','To'])
            return order, edges
library(pcalg)
library(EqVarDAG)
library(bnlearn)
source("NPVAR.R")
source("NPVAR_utils.R")

# In this demo, we will use GES, MMHC, EqVar_TD, EqVar_HD, NPVAR to learn sachs data

edge_stat <- function(real,find) {
  real_edge_len = nrow(real)
  esti_edge_len = nrow(find)
  rbinddf = rbind(real,find)
  True_edge_number = sum(duplicated(rbinddf))
  findflip = find[,c(2,1)]
  rbindfldf = rbind(real,findflip)
  Flip_edge_number = sum(duplicated(rbindfldf))
  TPR=round(True_edge_number/real_edge_len*100,2)
  FDR=round((1-True_edge_number/esti_edge_len)*100,2)
  SHD=(esti_edge_len-True_edge_number-Flip_edge_number) + (real_edge_len-True_edge_number)
  return (c(TPR,FDR,SHD,Flip_edge_number,esti_edge_len))
}

sachs = read.csv(paste0(getwd(),'/sachs.csv'))
sachs = sachs[,2:12]
dnames = colnames(sachs)
sachsm = as.matrix(sachs)
gstandard = c("Plcg","PIP3","Plcg","PIP2","PIP3","PIP2","Plcg","PKC","PIP2","PKC","PIP3","Akt",
              "PKC","Mek","PKC","Raf","PKC","PKA","PKC","Jnk","PKC","P38","PKA","Raf","PKA","Mek",
              "PKA","Erk","PKA","Akt","PKA","Jnk","PKA","P38","Raf","Mek","Mek","Erk","Erk","Akt")
goldstandard = matrix(gstandard, ncol=2, byrow=TRUE)


## GES

score = new("GaussL0penObsScore", sachsm)
ges.fit = ges(score)
am = as(as(ges.fit$essgraph,"graphNEL"),"Matrix")
sm = summary(am)
find = cbind(dnames[sm$i],dnames[sm$j])
edge_stat(goldstandard,find)
# TPR,FDR,SHD,Flip_edge_number,edges = (55.0 72.5 29.0  9.0 40.0)


## MMHC

mmhcresult = mmhc(sachs)
find = mmhcresult$arcs
edge_stat(goldstandard,find)
# TPR,FDR,SHD,Flip_edge_number,edges = (40.00 63.64 24.00  2.00 22.00)


## EqVar_TD

etdresult = EqVarDAG_TD(sachsm)
findindex = which(etdresult$adj==TRUE, arr.ind = TRUE)
find = matrix(dnames[findindex],ncol=2)
edge_stat(goldstandard,find)
# TPR,FDR,SHD,Flip_edge_number,edges = (30.00 87.23 43.00 12.00 47.00)


## EqVar_BU

eburesult = EqVarDAG_BU(sachsm)
findindex = which(eburesult$adj==TRUE, arr.ind = TRUE)
find = matrix(dnames[findindex],ncol=2)
edge_stat(goldstandard,find)
# TPR,FDR,SHD,Flip_edge_number,edges = (25.0 89.8 44.0 15.0 49.0)


## NPVAR

npresult = NPVAR(sachsm)
est = prune(sachsm, npresult)
findindex = which(est==TRUE, arr.ind = TRUE)
find = matrix(dnames[findindex],ncol=2)
edge_stat(goldstandard,find)
# TPR,FDR,SHD,Flip_edge_number,edges = (40.00 82.22 40.00  9.00 45.00)


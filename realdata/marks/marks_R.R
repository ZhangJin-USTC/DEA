library(EqVarDAG)
source("NPVAR.R")
source("NPVAR_utils.R")

marks = read.csv(paste0(getwd(),'/marks.csv'))
marks = marks[,2:6]
dnames = colnames(marks)
marksm = as.matrix(marks)

## EqVar_TD

etdresult = EqVarDAG_TD(marksm)
findindex = which(etdresult$adj==TRUE, arr.ind = TRUE)
find = matrix(dnames[findindex],ncol=2)
find

# "VECT" "MECH"
# "ALG"  "MECH"
# "ALG"  "VECT"
# "ALG"  "ANL" 
# "ALG"  "STAT"
# "ANL"  "STAT"


## EqVar_BU

eburesult = EqVarDAG_BU(marksm)
findindex = which(eburesult$adj==TRUE, arr.ind = TRUE)
find = matrix(dnames[findindex],ncol=2)
find

# "VECT" "MECH"
# "ALG"  "MECH"
# "ALG"  "VECT"
# "ALG"  "ANL" 
# "ALG"  "STAT"
# "ANL"  "STAT"  (same as EqVar_TD)


## NPVAR

npresult = NPVAR(marksm)
est = prune(marksm, npresult)
findindex = which(est==TRUE, arr.ind = TRUE)
find = matrix(dnames[findindex],ncol=2)
find

# "ALG" "ANL"


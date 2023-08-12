library(pcalg)
library(EqVarDAG)
library(bnlearn)
library(ggplot2)

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
  return (c(TPR,FDR,SHD))
}

path = paste0(getwd(),'/data')

# settings=rbind(c(30,30),c(30,100),c(30,300),c(100,50),c(100,200),c(100,500),c(500,100),c(500,500),c(500,1000))
settings = rbind(c(30,30),c(30,100),c(30,300))
# gtype=c("ER2","ER5","BA2","BA5")
gtype = c("ER2")
errtype = c("Uniform","Gumbel","t","Laplace")

for (err in errtype) {
for (i_gt in gtype)
  for (i_se in 1:3) {
    p=settings[i_se,1]
    n=settings[i_se,2]
    res1 = matrix(0,nrow=100,ncol=4)
    res2 = matrix(0,nrow=100,ncol=4)
    res3 = matrix(0,nrow=100,ncol=4)
    res4 = matrix(0,nrow=100,ncol=4)
    res5 = matrix(0,nrow=100,ncol=4)
    for (i in 1:100) {
      data = read.csv(paste0(path,'/',i_gt,'/',err,'/n',n,'/',i_gt,'_',err,'_p',p,'_n',n,'_',i-1,'.csv'))
      ginfo = read.csv(paste0(path,'/',i_gt,'/graph/',i_gt,'_p',p,'_',i-1,'.csv'))
      real=as.matrix(ginfo)[,2:3]
      data=data[,2:(p+1)]
      dnames = colnames(data)
      datam = as.matrix(data)
      
      # PC
      start=Sys.time()
      pc.fit <- pc(suffStat=list(C=cor(data),n=30),indepTest = gaussCItest,labels=dnames,alpha=0.5) # default alpha=0.05, which choose too few edges!
      end=Sys.time()
      ww=pc.fit@graph@edgeL
      find1=matrix(data=NA,nrow=0,ncol=2)
      for (v in 1:p) {
        if (length(ww[[v]]$edges)>0) {
          leg = length(ww[[v]]$edges)
          for (vv in 1:leg)
            find1 = rbind(find1,c(names(ww)[v], dnames[ww[[v]]$edges][vv]))
        }
      }
      rowcount = nrow(find1)
      find2 = cbind(find1[,2],find1[,1])
      frev = rbind(find2,find1)
      fdup = frev[duplicated(frev),]
      if (nrow(fdup)==0) {
        find = find1
      } else {
        indexsave = NULL
        for (ii in 1:nrow(fdup)) {
          if (as.numeric(substring(fdup[ii,1],2)) > as.numeric(substring(fdup[ii,2],2)))
            indexsave = c(indexsave,ii)
        }
        fdup = fdup[-indexsave,]
        fnotdup = frev[!duplicated(frev),]
        fnotdup = fnotdup[-(1:rowcount),]
        find = rbind(fnotdup,fdup)    # automatically change the bi-orient edges to the correct orient, by the index from small to large. We can find the closest MEC
      }
      result = edge_stat(real,find)
      res1[i,1] = round(as.numeric(difftime(end,start,units = 'secs')),2)
      res1[i,2:4] = result
      
      # GES
      start=Sys.time()
      score <- new("GaussL0penObsScore", datam)
      ges.fit <- ges(score)
      end=Sys.time()
      am=as(as(ges.fit$essgraph,"graphNEL"),"Matrix")
      www=summary(am)
      find=cbind(dnames[www$i],dnames[www$j])
      result = edge_stat(real,find)
      res2[i,1] = round(as.numeric(difftime(end,start,units = 'secs')),2)
      res2[i,2:4] = result
      
      # MMHC
      start=Sys.time()
      ppp = mmhc(data)
      end=Sys.time()
      find = ppp$arcs
      result = edge_stat(real,find)
      res3[i,1] = round(as.numeric(difftime(end,start,units = 'secs')),2)
      res3[i,2:4] = result
      
      # EqVar_HD_TD
      start=Sys.time()
      w=EqVarDAG_HD_TD(datam)  # default J=3
      end=Sys.time()
      findindex = which(w$adj==TRUE, arr.ind = TRUE)
      find = matrix(dnames[findindex],ncol=2)
      result = edge_stat(real,find)
      res4[i,1] = round(as.numeric(difftime(end,start,units = 'secs')),2)
      res4[i,2:4] = result
      
      # EqVar_HD_CLIME
      start=Sys.time()
      w_bu = EqVarDAG_HD_CLIME(datam)
      end=Sys.time()
      findindex = which(w_bu$adj==TRUE, arr.ind = TRUE)
      find = matrix(dnames[findindex],ncol=2)
      result = edge_stat(real,find)
      res5[i,1] = round(as.numeric(difftime(end,start,units = 'secs')),2)
      res5[i,2:4] = result
      
      write.csv(res1,paste0(path,'/',i_gt,'/',err,'/n',n,'/Result_PC.csv'))
      write.csv(res2,paste0(path,'/',i_gt,'/',err,'/n',n,'/Result_GES.csv'))
      write.csv(res3,paste0(path,'/',i_gt,'/',err,'/n',n,'/Result_MMHC.csv'))
      write.csv(res4,paste0(path,'/',i_gt,'/',err,'/n',n,'/Result_EqVar.csv'))
      write.csv(res5,paste0(path,'/',i_gt,'/',err,'/n',n,'/Result_CLIME.csv'))
      print(paste0(i_gt,' ',n,' ',i,' ',err))
    }
  }
}
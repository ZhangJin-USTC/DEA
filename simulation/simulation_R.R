library(pcalg)
library(EqVarDAG)
library(bnlearn)
library(ggplot2)
source("NPVAR.R")
source("NPVAR_utils.R")

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
gtype = c("ER2","ER5")

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
      data = read.csv(paste0(path,'/',i_gt,'/p',p,'/n',n,'/',i_gt,'_p',p,'_n',n,'_',i-1,'.csv'))
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
      
      write.csv(res1,paste0(path,'/',i_gt,'/p',p,'/n',n,'/Result_PC.csv'))
      write.csv(res2,paste0(path,'/',i_gt,'/p',p,'/n',n,'/Result_GES.csv'))
      write.csv(res3,paste0(path,'/',i_gt,'/p',p,'/n',n,'/Result_MMHC.csv'))
      write.csv(res4,paste0(path,'/',i_gt,'/p',p,'/n',n,'/Result_EqVar.csv'))
      write.csv(res5,paste0(path,'/',i_gt,'/p',p,'/n',n,'/Result_CLIME.csv'))
      print(paste0(i_gt,' ',n,' ',i))
    }
  }

# NPVAR
for (i_gt in gtype) {
  p=30
  n=300
  res = matrix(0,nrow=100,ncol=4)
  for (i in 1:100) {
    data = read.csv(paste0(path,'/',i_gt,'/p',p,'/n',n,'/',i_gt,'_p',p,'_n',n,'_',i-1,'.csv'))
    ginfo = read.csv(paste0(path,'/',i_gt,'/graph/',i_gt,'_p',p,'_',i-1,'.csv'))
    real=as.matrix(ginfo)[,2:3]
    data=data[,2:(p+1)]
    dnames = colnames(data)
    datam = as.matrix(data)
    start=Sys.time()
    npresult = NPVAR(datam)
    est = prune(datam, npresult)
    end=Sys.time()
    findindex = which(est==TRUE, arr.ind = TRUE)
    find = matrix(dnames[findindex],ncol=2)
    result = edge_stat(real,find)
    res[i,1] = round(as.numeric(difftime(end,start,units = 'secs')),2)
    res[i,2:4] = result
    write.csv(res,paste0(path,'/',i_gt,'/p',p,'/n',n,'/Result_NPVAR.csv'))
    print(paste0(i,' ',,' '))
  }
}


# settings = rbind(c(30,30),c(30,100),c(30,300))
# gtype=c("ER2","ER5","BA2","BA5")
settings = rbind(c(100,50),c(100,200),c(100,500))
gtype = c("ER2","ER5","BA2","BA5")

results = data.frame()
mthds = c("EqVar","CLIME","PC","GES","MMHC")

for (i_gt in gtype)
    for (i_se in 1:3) {
      
      mds="abessdag"
      p=settings[i_se,1]
      n=settings[i_se,2]
      xx = read.csv(paste0(path,'/',i_gt,'/p',p,'/n',n,'/Result_',mds,'.csv'))
      aha = c(mds,i_gt,p,n,
              paste0(format(round(mean(xx$Time),2),nsmall=2),'(',format(round(sd(xx$Time),2),nsmall=2),')'),
              paste0(format(round(mean(xx$TPR),1),nsmall=1),'(',format(round(sd(xx$TPR),1),nsmall=1),')'),
              paste0(format(round(mean(xx$FDR),1),nsmall=1),'(',format(round(sd(xx$FDR),1),nsmall=1),')'),
              paste0(format(round(mean(xx$SHD),1),nsmall=1),'(',format(round(sd(xx$SHD),1),nsmall=1),')'))
      results = rbind(results,aha)
      
      for (mds in mthds) {
      p=settings[i_se,1]
      n=settings[i_se,2]
      for (mds in mthds) {
      xx = read.csv(paste0(path,'/',i_gt,'/p',p,'/n',n,'/Result_',mds,'.csv'))
      xx = xx[1:20,]
      aha = c(mds,i_gt,p,n,
              paste0(format(round(mean(xx$V1),2),nsmall=2),'(',format(round(sd(xx$V1),2),nsmall=2),')'),
              paste0(format(round(mean(xx$V2),1),nsmall=1),'(',format(round(sd(xx$V2),1),nsmall=1),')'),
              paste0(format(round(mean(xx$V3),1),nsmall=1),'(',format(round(sd(xx$V3),1),nsmall=1),')'),
              paste0(format(round(mean(xx$V4),1),nsmall=1),'(',format(round(sd(xx$V4),1),nsmall=1),')'))
      results = rbind(results,aha)
      }
      
      
  }

# mthds = c("abessdag","notears")
mthds="abessdag"
for (i_gt in gtype)
  for (mds in mthds)  {
    for (i_se in 1:3) {
      p=settings[i_se,1]
      n=settings[i_se,2]
      xx = read.csv(paste0(path,'/',i_gt,'/p',p,'/n',n,'/Result_',mds,'.csv'))
      aha = c(mds,i_gt,p,n,
              paste0(format(round(mean(xx$Time),2),nsmall=2),'(',format(round(sd(xx$Time),2),nsmall=2),')'),
              paste0(format(round(mean(xx$TPR),1),nsmall=1),'(',format(round(sd(xx$TPR),1),nsmall=1),')'),
              paste0(format(round(mean(xx$FDR),1),nsmall=1),'(',format(round(sd(xx$FDR),1),nsmall=1),')'),
              paste0(format(round(mean(xx$SHD),1),nsmall=1),'(',format(round(sd(xx$SHD),1),nsmall=1),')'))
      results = rbind(results,aha)
    }
  }

mds="NPVAR"
for (i_gt in gtype) {
  p=30
  n=300
  xx = read.csv(paste0(path,'/',i_gt,'/p',p,'/n',n,'/Result_',mds,'.csv'))
  aha = c(mds,i_gt,p,n,
          paste0(format(round(mean(xx$V1),2),nsmall=2),'(',format(round(sd(xx$V1),2),nsmall=2),')'),
          paste0(format(round(mean(xx$V2),1),nsmall=1),'(',format(round(sd(xx$V2),1),nsmall=1),')'),
          paste0(format(round(mean(xx$V3),1),nsmall=1),'(',format(round(sd(xx$V3),1),nsmall=1),')'),
          paste0(format(round(mean(xx$V4),1),nsmall=1),'(',format(round(sd(xx$V4),1),nsmall=1),')'))
  results = rbind(results,aha)
}

colnames(results)=c("Method","Graph","p","n","Time","TPR","FDR","SHD")
pathr = paste0(getwd(),'/result')
write.csv(results,paste0(pathr,'/Results_SF_p30.csv'))


### Graph plot

## ER2,TPR

df = data.frame()
i_gt = "ER2"
mthds = c("PC","GES","MMHC","EqVar","CLIME")
for (mds in mthds)
  for (i_se in 1:3) {
    p=settings[i_se,1]
    n=settings[i_se,2]
    xx = read.csv(paste0(path,'/',i_gt,'/p',p,'/n',n,'/Result_',mds,'.csv'))
    aha = c(mds,n,mean(xx$V2),sd(xx$V2))
    df = rbind(df,aha)
  }

mds="abessdag"
for (i_se in 1:3) {
  p=settings[i_se,1]
  n=settings[i_se,2]
  xx = read.csv(paste0(path,'/',i_gt,'/p',p,'/n',n,'/Result_',mds,'.csv'))
  aha = c("ABESS-DAG",n,mean(xx$TPR),sd(xx$TPR))
  df = rbind(df,aha)
}

mds="notears"
for (i_se in 1:3) {
  p=settings[i_se,1]
  n=settings[i_se,2]
  xx = read.csv(paste0(path,'/',i_gt,'/p',p,'/n',n,'/Result_',mds,'.csv'))
  aha = c("NOTEARS",n,mean(xx$TPR),sd(xx$TPR))
  df = rbind(df,aha)
}

colnames(df)=c("Method","n","TPR","sd")
df$TPR = as.numeric(df$TPR)
df$sd = as.numeric(df$sd)
df$n = factor(df$n,levels=c("30","100","300"))

p1 <- ggplot(df, aes(x=n, y=TPR, group=Method, color=Method)) + 
 # geom_errorbar(aes(ymin=TPR-sd, ymax=TPR+sd), width=.1,position=position_dodge(0.05)) +
  geom_line(size=1) + geom_point(size=2)+
  scale_color_brewer(palette="Paired")+theme_minimal()+
  ggtitle("TPR on ER2 graphs, p=30")

## ER2, FDR

df = data.frame()
i_gt = "ER2"
mthds = c("PC","GES","MMHC","EqVar","CLIME")
for (mds in mthds)
  for (i_se in 1:3) {
    p=settings[i_se,1]
    n=settings[i_se,2]
    xx = read.csv(paste0(path,'/',i_gt,'/p',p,'/n',n,'/Result_',mds,'.csv'))
    aha = c(mds,n,mean(xx$V3),sd(xx$V3))
    df = rbind(df,aha)
  }

mds="abessdag"
for (i_se in 1:3) {
  p=settings[i_se,1]
  n=settings[i_se,2]
  xx = read.csv(paste0(path,'/',i_gt,'/p',p,'/n',n,'/Result_',mds,'.csv'))
  aha = c("ABESS-DAG",n,mean(xx$FDR),sd(xx$FDR))
  df = rbind(df,aha)
}

mds="notears"
for (i_se in 1:3) {
  p=settings[i_se,1]
  n=settings[i_se,2]
  xx = read.csv(paste0(path,'/',i_gt,'/p',p,'/n',n,'/Result_',mds,'.csv'))
  aha = c("NOTEARS",n,mean(xx$FDR),sd(xx$FDR))
  df = rbind(df,aha)
}

colnames(df)=c("Method","n","FDR","sd")
df$FDR = as.numeric(df$FDR)
df$sd = as.numeric(df$sd)
df$n = factor(df$n,levels=c("30","100","300"))

p2 <- ggplot(df, aes(x=n, y=FDR, group=Method, color=Method)) + 
  # geom_errorbar(aes(ymin=TPR-sd, ymax=TPR+sd), width=.1,position=position_dodge(0.05)) +
  geom_line(size=1) + geom_point(size=2)+
  scale_color_brewer(palette="Paired")+theme_minimal()+
  ggtitle("FDR on ER2 graphs, p=30")


## ER2,SHD

df = data.frame()
i_gt = "ER2"
mthds = c("PC","GES","MMHC","EqVar","CLIME")
for (mds in mthds)
  for (i_se in 1:3) {
    p=settings[i_se,1]
    n=settings[i_se,2]
    xx = read.csv(paste0(path,'/',i_gt,'/p',p,'/n',n,'/Result_',mds,'.csv'))
    aha = c(mds,n,mean(xx$V4),sd(xx$V4))
    df = rbind(df,aha)
  }

mds="abessdag"
for (i_se in 1:3) {
  p=settings[i_se,1]
  n=settings[i_se,2]
  xx = read.csv(paste0(path,'/',i_gt,'/p',p,'/n',n,'/Result_',mds,'.csv'))
  aha = c("ABESS-DAG",n,mean(xx$SHD),sd(xx$SHD))
  df = rbind(df,aha)
}

mds="notears"
for (i_se in 1:3) {
  p=settings[i_se,1]
  n=settings[i_se,2]
  xx = read.csv(paste0(path,'/',i_gt,'/p',p,'/n',n,'/Result_',mds,'.csv'))
  aha = c("NOTEARS",n,mean(xx$SHD),sd(xx$SHD))
  df = rbind(df,aha)
}

colnames(df)=c("Method","n","SHD","sd")
df$SHD = as.numeric(df$SHD)
df$sd = as.numeric(df$sd)
df$n = factor(df$n,levels=c("30","100","300"))

p3 <- ggplot(df, aes(x=n, y=SHD, group=Method, color=Method)) + 
  # geom_errorbar(aes(ymin=TPR-sd, ymax=TPR+sd), width=.1,position=position_dodge(0.05)) +
  geom_line(size=1) + geom_point(size=2)+
  scale_color_brewer(palette="Paired")+theme_minimal()+
  ggtitle("SHD on ER2 graphs, p=30")

grid.arrange(p1,p2,p3,nrow=1)



## ER5 graph

## TPR

df = data.frame()
i_gt = "ER5"
mthds = c("PC","GES","MMHC","EqVar","CLIME")
for (mds in mthds)
  for (i_se in 1:3) {
    p=settings[i_se,1]
    n=settings[i_se,2]
    xx = read.csv(paste0(path,'/',i_gt,'/p',p,'/n',n,'/Result_',mds,'.csv'))
    aha = c(mds,n,mean(xx$V2),sd(xx$V2))
    df = rbind(df,aha)
  }

mds="abessdag"
for (i_se in 1:3) {
  p=settings[i_se,1]
  n=settings[i_se,2]
  xx = read.csv(paste0(path,'/',i_gt,'/p',p,'/n',n,'/Result_',mds,'.csv'))
  aha = c("ABESS-DAG",n,mean(xx$TPR),sd(xx$TPR))
  df = rbind(df,aha)
}

mds="notears"
for (i_se in 1:3) {
  p=settings[i_se,1]
  n=settings[i_se,2]
  xx = read.csv(paste0(path,'/',i_gt,'/p',p,'/n',n,'/Result_',mds,'.csv'))
  aha = c("NOTEARS",n,mean(xx$TPR),sd(xx$TPR))
  df = rbind(df,aha)
}

colnames(df)=c("Method","n","TPR","sd")
df$TPR = as.numeric(df$TPR)
df$sd = as.numeric(df$sd)
df$n = factor(df$n,levels=c("30","100","300"))

p1 <- ggplot(df, aes(x=n, y=TPR, group=Method, color=Method)) + 
  # geom_errorbar(aes(ymin=TPR-sd, ymax=TPR+sd), width=.1,position=position_dodge(0.05)) +
  geom_line(size=1) + geom_point(size=2)+
  scale_color_brewer(palette="Paired")+theme_minimal()+
  ggtitle("TPR on ER5 graphs, p=30")

df = data.frame()
i_gt = "ER5"
mthds = c("PC","GES","MMHC","EqVar","CLIME")
for (mds in mthds)
  for (i_se in 1:3) {
    p=settings[i_se,1]
    n=settings[i_se,2]
    xx = read.csv(paste0(path,'/',i_gt,'/p',p,'/n',n,'/Result_',mds,'.csv'))
    aha = c(mds,n,mean(xx$V3),sd(xx$V3))
    df = rbind(df,aha)
  }

mds="abessdag"
for (i_se in 1:3) {
  p=settings[i_se,1]
  n=settings[i_se,2]
  xx = read.csv(paste0(path,'/',i_gt,'/p',p,'/n',n,'/Result_',mds,'.csv'))
  aha = c("ABESS-DAG",n,mean(xx$FDR),sd(xx$FDR))
  df = rbind(df,aha)
}

mds="notears"
for (i_se in 1:3) {
  p=settings[i_se,1]
  n=settings[i_se,2]
  xx = read.csv(paste0(path,'/',i_gt,'/p',p,'/n',n,'/Result_',mds,'.csv'))
  aha = c("NOTEARS",n,mean(xx$FDR),sd(xx$FDR))
  df = rbind(df,aha)
}

colnames(df)=c("Method","n","FDR","sd")
df$FDR = as.numeric(df$FDR)
df$sd = as.numeric(df$sd)
df$n = factor(df$n,levels=c("30","100","300"))

p2 <- ggplot(df, aes(x=n, y=FDR, group=Method, color=Method)) + 
  # geom_errorbar(aes(ymin=TPR-sd, ymax=TPR+sd), width=.1,position=position_dodge(0.05)) +
  geom_line(size=1) + geom_point(size=2)+
  scale_color_brewer(palette="Paired")+theme_minimal()+
  ggtitle("FDR on ER5 graphs, p=30")


## SHD

df = data.frame()
i_gt = "ER5"
mthds = c("PC","GES","MMHC","EqVar","CLIME")
for (mds in mthds)
  for (i_se in 1:3) {
    p=settings[i_se,1]
    n=settings[i_se,2]
    xx = read.csv(paste0(path,'/',i_gt,'/p',p,'/n',n,'/Result_',mds,'.csv'))
    aha = c(mds,n,mean(xx$V4),sd(xx$V4))
    df = rbind(df,aha)
  }

mds="abessdag"
for (i_se in 1:3) {
  p=settings[i_se,1]
  n=settings[i_se,2]
  xx = read.csv(paste0(path,'/',i_gt,'/p',p,'/n',n,'/Result_',mds,'.csv'))
  aha = c("ABESS-DAG",n,mean(xx$SHD),sd(xx$SHD))
  df = rbind(df,aha)
}

mds="notears"
for (i_se in 1:3) {
  p=settings[i_se,1]
  n=settings[i_se,2]
  xx = read.csv(paste0(path,'/',i_gt,'/p',p,'/n',n,'/Result_',mds,'.csv'))
  aha = c("NOTEARS",n,mean(xx$SHD),sd(xx$SHD))
  df = rbind(df,aha)
}

colnames(df)=c("Method","n","SHD","sd")
df$SHD = as.numeric(df$SHD)
df$sd = as.numeric(df$sd)
df$n = factor(df$n,levels=c("30","100","300"))

p3 <- ggplot(df, aes(x=n, y=SHD, group=Method, color=Method)) + 
  # geom_errorbar(aes(ymin=TPR-sd, ymax=TPR+sd), width=.1,position=position_dodge(0.05)) +
  geom_line(size=1) + geom_point(size=2)+
  scale_color_brewer(palette="Paired")+theme_minimal()+
  ggtitle("SHD on ER5 graphs, p=30")

grid.arrange(p1,p2,p3,nrow=1)









# p=300 Result summary

settings = rbind(c(300,100),c(300,300),c(300,1000))
gtype = c("ER2","ER5","BA2","BA5")

results = data.frame()
mthds = c("EqVar","CLIME","PC","GES","MMHC")

for (i_gt in gtype)
  for (i_se in 1:3) {
    
    mds="abessdag"
    p=settings[i_se,1]
    n=settings[i_se,2]
    xx = read.csv(paste0(path,'/',i_gt,'/p',p,'/n',n,'/Result_',mds,'.csv'))
    aha = c(mds,i_gt,p,n,
            paste0(format(round(mean(xx$Time),2),nsmall=2),'(',format(round(sd(xx$Time),2),nsmall=2),')'),
            paste0(format(round(mean(xx$TPR),1),nsmall=1),'(',format(round(sd(xx$TPR),1),nsmall=1),')'),
            paste0(format(round(mean(xx$FDR),1),nsmall=1),'(',format(round(sd(xx$FDR),1),nsmall=1),')'),
            paste0(format(round(mean(xx$SHD),1),nsmall=1),'(',format(round(sd(xx$SHD),1),nsmall=1),')'))
    results = rbind(results,aha)
    
    for (mds in mthds) {
      p=settings[i_se,1]
      n=settings[i_se,2]
      for (mds in mthds) {
        xx = read.csv(paste0(path,'/',i_gt,'/p',p,'/n',n,'/Result_',mds,'.csv'))
        xx = xx[1:20,]
        aha = c(mds,i_gt,p,n,
                paste0(format(round(mean(xx$V1),2),nsmall=2),'(',format(round(sd(xx$V1),2),nsmall=2),')'),
                paste0(format(round(mean(xx$V2),1),nsmall=1),'(',format(round(sd(xx$V2),1),nsmall=1),')'),
                paste0(format(round(mean(xx$V3),1),nsmall=1),'(',format(round(sd(xx$V3),1),nsmall=1),')'),
                paste0(format(round(mean(xx$V4),1),nsmall=1),'(',format(round(sd(xx$V4),1),nsmall=1),')'))
        results = rbind(results,aha)
      }
    }
}

    
    colnames(results)=c("Method","Graph","p","n","Time","TPR","FDR","SHD")
    pathr = paste0(getwd(),'/result')
    write.csv(results,paste0(pathr,'/Results_SF_p30.csv'))
    
    

    
    
# Heteroskedastic case (Section 5.3)

settings = rbind(c(30,30),c(30,100),c(30,300))
gtype = c("ER2","ER5","BA2","BA5")

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
      data = read.csv(paste0(path,'/',i_gt,'/p',p,'_hetero/n',n,'/',i_gt,'_p',p,'_n',n,'_',i-1,'.csv'))
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
      
      write.csv(res1,paste0(path,'/',i_gt,'/p',p,'_hetero/n',n,'/Result_PC.csv'))
      write.csv(res2,paste0(path,'/',i_gt,'/p',p,'_hetero/n',n,'/Result_GES.csv'))
      write.csv(res3,paste0(path,'/',i_gt,'/p',p,'_hetero/n',n,'/Result_MMHC.csv'))
      write.csv(res4,paste0(path,'/',i_gt,'/p',p,'_hetero/n',n,'/Result_EqVar.csv'))
      write.csv(res5,paste0(path,'/',i_gt,'/p',p,'_hetero/n',n,'/Result_CLIME.csv'))
      print(paste0(i_gt,' ',n,' ',i))
    }
  }


# Heteroskedastic case summary

settings = rbind(c(30,30),c(30,100),c(30,300))
gtype = c("ER2","ER5")

results = data.frame()
mthds = c("EqVar","CLIME","PC","GES","MMHC")
mthdspython = c("abessdag","notears","pabessdag_20","pabessdag_50","pabessdag_80")

for (i_gt in gtype)
  for (i_se in 1:3) {
    
    for (mds in mthdspython) {
      p=settings[i_se,1]
      n=settings[i_se,2]
      xx = read.csv(paste0(path,'/',i_gt,'/p',p,'_hetero/n',n,'/Result_',mds,'.csv'))
      aha = c(mds,i_gt,p,n,
              paste0(format(round(mean(xx$Time),2),nsmall=2),'(',format(round(sd(xx$Time),2),nsmall=2),')'),
              paste0(format(round(mean(xx$TPR),1),nsmall=1),'(',format(round(sd(xx$TPR),1),nsmall=1),')'),
              paste0(format(round(mean(xx$FDR),1),nsmall=1),'(',format(round(sd(xx$FDR),1),nsmall=1),')'),
              paste0(format(round(mean(xx$SHD),1),nsmall=1),'(',format(round(sd(xx$SHD),1),nsmall=1),')'))
      results = rbind(results,aha)
    }
    
    for (mds in mthds) {
      p=settings[i_se,1]
      n=settings[i_se,2]
      xx = read.csv(paste0(path,'/',i_gt,'/p',p,'_hetero/n',n,'/Result_',mds,'.csv'))
      aha = c(mds,i_gt,p,n,
                paste0(format(round(mean(xx$V1),2),nsmall=2),'(',format(round(sd(xx$V1),2),nsmall=2),')'),
                paste0(format(round(mean(xx$V2),1),nsmall=1),'(',format(round(sd(xx$V2),1),nsmall=1),')'),
                paste0(format(round(mean(xx$V3),1),nsmall=1),'(',format(round(sd(xx$V3),1),nsmall=1),')'),
                paste0(format(round(mean(xx$V4),1),nsmall=1),'(',format(round(sd(xx$V4),1),nsmall=1),')'))
      results = rbind(results,aha)
    }
  }


colnames(results)=c("Method","Graph","p","n","Time","TPR","FDR","SHD")
pathr = paste0(getwd(),'/result')
write.csv(results,paste0(pathr,'/Results_hetero_p30.csv'))


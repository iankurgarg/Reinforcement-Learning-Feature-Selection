library(stats)

data = read.csv('data/categorical_data.csv')

cols = colnames(data)

V = matrix(0.0, nrow=24, ncol=24)

for (i in 1:length(cols)) {
  for (j in 1:length(cols)) {
    tbl = table(data[,cols[i]], data[,cols[j]])
    chi2 = chisq.test(tbl, correct = F)
    V[i,j] = sqrt(chi2$statistic/sum(tbl))
  }
}


require('FactoMineR')
data = read.csv('data/MDP_Original_data2.csv')

x = FAMD(data, ncp = 8)

write.csv(x$ind$coord, file='data/FAMD_features.csv', row.names = FALSE)

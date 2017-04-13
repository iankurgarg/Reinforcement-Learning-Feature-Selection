library(stats)
require('FactoMineR')

data = read.csv('data/MDP_Original_data2.csv')

x = FAMD(data, ncp = 8)

write.csv(x$ind$coord, file='data/FAMD_features.csv', row.names = FALSE)

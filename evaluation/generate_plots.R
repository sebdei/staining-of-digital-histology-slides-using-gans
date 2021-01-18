require(dplyr)
require(purrr)
require(ggplot2)
library("scales") 

setwd("./projects/masterarbeit/evaluation/")
csv_files = list.files("./", pattern=".csv")

plot_results = function (file_name) {
  df = read.csv(file_name)
  
  df = df %>% 
    mutate(precision=TP / (TP + FP)) %>%
    mutate(recall=TP / (TP + FN)) %>%
    mutate(f1=(2 * precision * recall) / (precision + recall))
  
  color = c(biobank1="#F8766D", biobank2="#A3A500", biobank3="#00BF7D", biobank4="#00B0F6", biobank5="#E76BF3")
  name = strsplit(file_name, split='.', fixed=TRUE)[[1]][1]  
  
  ggplot(df, aes(x=epoch, y=f1, color=dataset)) +
    geom_line() +
    geom_point() +
    ggtitle(name) +
    scale_color_manual(values=color)
  
  ggsave(paste("./plots/", name, "_f1.jpg", sep=""), plot=last_plot())
  
  ggplot(df, aes(x=epoch, y=ssim, color=dataset)) +
    geom_line() +
    geom_point() +
    ggtitle(name) +
    scale_color_manual(values=color)
  
  ggsave(paste("./plots/", name, "_ssim.jpg", sep=""), plot=last_plot())
}

sapply(csv_files, plot_results)

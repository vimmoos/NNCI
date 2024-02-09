library(tidyverse)
options(pillar.subtle = FALSE)
library(ggplot2)


dat <- read.csv("res.csv")  %>% as_tibble

th = theme(
  axis.text = element_text(size = 25),     # Adjust the size of axis ticks
  axis.title = element_text(size = 25),    # Adjust the size of axis labels
  legend.title = element_text(size = 25),  # Adjust the size of legend title
  legend.text = element_text(size = 25)    # Adjust the size of legend text
)

sum_data <- dat  %>%
  filter(c == 0) %>%
  group_by(alpha) %>%
  summarise(prob = mean(solutions)/mean(n_dataset))

ggplot(sum_data,aes(alpha,prob)) +
    geom_line()+
    geom_point() + ylab("Probability of success") + xlab("Alpha") + th

ggsave("images/simple.png")

sum_data <- dat %>% group_by(alpha,c) %>% summarise(prob = mean(solutions)/mean(n_dataset))

ggplot(sum_data,aes(alpha,prob,color=c,group=c)) +
    geom_line()+
  geom_point() + ylab("Probability of success") + xlab("Alpha") + th

ggsave("images/withC.png",)

sum_data <- dat %>% group_by(alpha,N) %>% summarise(prob = mean(solutions)/mean(n_dataset))

ggplot(sum_data,aes(alpha,prob,color=N,group=N)) +
    geom_line()+
    geom_point() + ylab("Probability of success") + xlab("Alpha") +th

ggsave("images/withN.png")


sum_data <- dat %>% group_by(alpha,c) %>% summarise(error = mean(error_mean))

ggplot(sum_data,aes(alpha,error,color=c,group=c)) +
    geom_line()+
  geom_point() + ylab("Error mean") + xlab("Alpha") + th

ggsave("images/withC_error.png")

sum_data <- dat %>% group_by(alpha,c) %>% summarise(error = mean(error_std))

ggplot(sum_data,aes(alpha,error,color=c,group=c)) +
    geom_line()+
  geom_point() + ylab("Error std") + xlab("Alpha") + th

ggsave("images/withC_std.png")



ggplot(dat,aes(error_std,error_mean,color=c,group=c,fill=c,size=-alpha)) +
  geom_point() + ylab("Error mean") + xlab("Error std") + th

ggsave("images/mean_vs_std.png")

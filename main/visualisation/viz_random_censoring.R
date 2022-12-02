##------------------------------------------------##
##  Author:     
##  University: 
##  Title:      Visualisation of censoring
##              KM and CKM estimates
##  Year:       2022
##  Run on:     Apple aarch64, R 4.1.2
##------------------------------------------------##
rm(list=ls())
project_dir = ''

setwd(project_dir)
library(tidyverse)
library(ggpattern)
extrafont::loadfonts(quiet = T)

global_theme <- function(){
  
  theme_minimal() %+replace%
    theme(
      text=element_text(family='Montserrat', size=14),
      axis.text = element_text(size=14), 
      plot.title = element_text(family='Montserrat SemiBold', size=18, hjust = 0.5),
      plot.subtitle = element_text(hjust = 0.5)
    )
}


#### Independent Censoring ####

data_km <- read_csv('data/random_censoring/estimates_kaplan_meier.csv')
data_ckm <- read_csv('data/random_censoring/estimates_conditional_km.csv') 
true_data <- read_csv('data/random_censoring/true_data.csv')
linear_ckm <- read_csv('data/random_censoring/estimates_unconditional_deep_km.csv')

# automatize aggregation
size = dim(data_ckm)[2] - 1

# Build rowwise average for ckm
data_ckm %>% 
  gather(individual_id, estimate, c(`0`:(size+1))) %>% 
  group_by(timeline) %>% 
  summarise(
    ckm_estimate_mean = mean(estimate), 
    ckm_estimate_median = median(estimate),
    ckm_estimate_std = sd(estimate)
  ) -> averaged_ckm

linear_ckm %>% 
  gather(individual_id, estimate, c(`0`:(size+1))) %>% 
  group_by(timeline) %>% 
  summarise(
    ckm_linear_mean = mean(estimate)
  ) -> averaged_ckm_linear

averaged_ckm_linear %>% 
  left_join(data_km) %>% 
  left_join(true_data) %>% {.->> joined_data} %>% 
  select(timeline, `CKM aggregated linear` = ckm_linear_mean,
         KM = KM_estimate, `True distribution` = survival) %>% 
  gather(Estimator, value, c(`CKM aggregated linear`:KM)) %>% 
  ggplot() + 
  geom_line(aes(x=timeline, y=value, color=Estimator, linetype=Estimator)) + 
  global_theme() + 
  scale_color_manual(values = c('darkgrey', 'black')) + 
  scale_linetype_manual(values=c('longdash', 'solid')) + 
  ggtitle('Survival curve estimation') + 
  ylab('S(t)') + 
  xlab('t')

ggsave(filename = 'outputs_images/random_censoring/linear_ckm_plot.png', 
       scale=0.3, height = 275, width = 700, units = 'mm', dpi = 600)

averaged_ckm %>% 
  left_join(averaged_ckm_linear) %>% 
  left_join(data_km) %>% 
  left_join(true_data) %>% {.->> joined_data} %>% 
  select(timeline, `CKM aggregated` = ckm_estimate_mean,
         KM = KM_estimate, `True distribution` = survival) %>% 
  gather(Estimator, value, c(`CKM aggregated`:KM)) %>% 
  ggplot() + 
  geom_line(aes(x=timeline, y=value, color=Estimator, linetype=Estimator)) + 
  global_theme() + 
  scale_color_manual(values = c('darkgrey', 'black')) + 
  scale_linetype_manual(values=c('longdash', 'solid')) + 
  ggtitle('Survival curve estimation') + 
  ylab('S(t)') + 
  xlab('t')

ggsave(filename = 'outputs_images/random_censoring/plot_km_ckm.png', 
       scale=0.3, height = 275, width = 700, units = 'mm', dpi = 600)

averaged_ckm %>% 
  left_join(averaged_ckm_linear) %>% 
  left_join(data_km) %>% 
  left_join(true_data) %>% {.->> joined_data} %>% 
  select(timeline, `CKM aggregated` = ckm_estimate_mean,
         KM = KM_estimate, `True distribution` = survival) %>% 
  gather(Estimator, value, c(`CKM aggregated`:`True distribution`)) %>% 
  ggplot() + 
  geom_line(aes(x=timeline, y=value, color=Estimator, linetype=Estimator)) + 
  global_theme() + 
  scale_color_manual(values = c('darkgrey', 'black', 'black')) + 
  scale_linetype_manual(values=c('longdash', 'solid', 'dotted')) + 
  ggtitle('Survival curve estimation') + 
  ylab('S(t)') + 
  xlab('t')

ggsave(filename = 'outputs_images/random_censoring/plot_all_estimators.png', 
       scale=0.3, height = 275, width = 700, units = 'mm', dpi = 600)

# Select indices from notebook
data_ckm %>% 
  select(timeline, `37`, `71`) %>% 
  left_join(data_km) %>% 
  select(timeline,
         `Low true (y=7)`=`37`, 
         `High true (y=27)`=`71`,
         KM = KM_estimate) %>% 
  gather(Estimation, value, c(`Low true (y=7)`:`KM`)) %>% 
  mutate(Estimation = factor(Estimation, 
                             levels = c("KM",
                                        "Low true (y=7)", 
                                        "High true (y=27)"))) %>% 
  ggplot() + 
  geom_line(aes(x=timeline, y=value, color=Estimation, linetype=Estimation)) + 
  global_theme() + 
  scale_color_manual(values = c('black', 'darkgrey', 'darkgrey')) + 
  scale_linetype_manual(values=c('solid', 'solid', 'longdash')) + 
  ggtitle('Individualized predictions') + 
  ylab('S(t)') + 
  xlab('t')

ggsave(filename = 'outputs_images/random_censoring/plot_individualized.png', 
       scale=0.3, height = 275, width = 700, units = 'mm', dpi = 600)


#### Dep. Censoring ####

## Linear dependence
rm(list=setdiff(ls(), "global_theme"))

data_km_linear <- read_csv('data/dependent_censoring/kaplan_meier_linearly_dep_censoring_1.csv')
data_ckm_linear <- read_csv('data/dependent_censoring/ckm_linear_dependent_censoring.csv') 
true_data_linear <- read_csv('data/dependent_censoring/true_data_linear_censoring.csv')

# automatize aggregation
size = dim(data_ckm_linear)[2] - 1

# Build rowwise average for ckm
data_ckm_linear %>% 
  gather(individual_id, estimate, c(`0`:(size+1))) %>% 
  group_by(timeline) %>% 
  summarise(
    ckm_estimate_mean = mean(estimate), 
    ckm_estimate_median = median(estimate),
    ckm_estimate_std = sd(estimate)
  ) -> averaged_ckm_linear


averaged_ckm_linear %>% 
  left_join(data_km_linear) %>% 
  left_join(true_data_linear) %>% {.->> joined_data} %>% 
  select(timeline, `CKM aggregated` = ckm_estimate_mean,
         KM = KM_estimate, `True distribution` = survival) %>% 
  gather(Estimator, value, c(`CKM aggregated`:`True distribution`)) %>% 
  ggplot() + 
  geom_line(aes(x=timeline, y=value, color=Estimator, linetype=Estimator)) + 
  global_theme() + 
  scale_color_manual(values = c('darkgrey', 'black', 'black')) + 
  scale_linetype_manual(values=c('longdash', 'solid', 'dotted')) + 
  ggtitle('Survival curve estimation', subtitle = 'With dependent censoring') + 
  ylab('S(t)') + 
  xlab('t')

ggsave(filename = 'outputs_images/dep_censoring/simple_dependent_censoring.png', 
       scale=0.3, height = 275, width = 700, units = 'mm', dpi = 600)


#### Robustness ####


data_ckm_dep <- read_csv('data/dependent_censoring/ckm_pred_heavy_censoring_small.csv')
km_heavy <- read_csv('data/dependent_censoring/km_heavy_censoring.csv')
true_data_heavy <- read_csv('data/dependent_censoring/heavy_censoring_true.csv')

data_ckm_heavy_larger = read_csv('data/dependent_censoring/ckm_pred_heavy_censoring_larger_model.csv')
ckm_manual_ <- read_csv('data/dependent_censoring/manually_corrected_version_dependent.csv')

size = 3000
size_large = 12000

data_ckm_dep %>% 
  gather(individual_id, estimate, c(`0`:(size+1))) %>% 
  group_by(timeline) %>% 
  summarise(
    ckm_estimate_mean = mean(estimate), 
    ckm_estimate_median = median(estimate),
    ckm_estimate_std = sd(estimate)
  ) -> averaged_ckm

ckm_manual_ %>% 
  gather(individual_id, estimate, c(`0`:(size+1))) %>% 
  group_by(timeline) %>% 
  summarise(
    ckm_estimate_mean_manual = mean(estimate), 
  ) -> averaged_ckm_manual

data_ckm_heavy_larger %>% 
  gather(individual_id, estimate, c(`0`:(size_large+1))) %>% 
  group_by(timeline) %>% 
  summarise(
    ckm_estimate_mean_large = mean(estimate), 
  ) -> averaged_ckm_large

averaged_ckm %>% 
  left_join(km_heavy) %>% 
  left_join(true_data_heavy) %>%  
  left_join(averaged_ckm_large) %>% 
  select(timeline, `CKM aggregate` = ckm_estimate_mean,
         `CKM aggregate large` = ckm_estimate_mean_large,
         KM_estimate,
         `True distribution` = survival) %>% 
  gather(Estimator, value, c(`CKM aggregate`:`True distribution`)) %>% 
  ggplot() + 
  geom_line(aes(x=timeline, y=value, color=Estimator, linetype=Estimator)) + 
  global_theme() + 
  scale_color_manual(values = c('grey', 'grey', 'black', 'black')) + 
  scale_linetype_manual(values=c('solid', 'dashed', 'solid', 'dotted')) + 
  ggtitle('Survival curve estimation') + 
  ylab('S(t)') + 
  xlab('t')

ggsave(filename = 'outputs_images/random_censoring/plot_larger_model_bias.png', 
       scale=0.3, height = 275, width = 700, units = 'mm', dpi = 600)

  
averaged_ckm_manual %>% 
  left_join(km_heavy) %>% 
  left_join(true_data_heavy) %>%  
  left_join(averaged_ckm) %>% 
  select(timeline,
         `CKM aggregate corrected` = ckm_estimate_mean_manual,
         `CKM aggregate uncorrected` = ckm_estimate_mean,
         KM = KM_estimate,
         `True distribution` = survival) %>% 
  gather(Estimator, value, c(`CKM aggregate corrected`:`True distribution`)) %>% 
  ggplot() + 
  geom_line(aes(x=timeline, y=value, color=Estimator, linetype=Estimator)) + 
  global_theme() + 
  scale_color_manual(values = c('grey', 'grey', 'black', 'black')) + 
  scale_linetype_manual(values=c('dashed', 'solid', 'solid', 'dotted')) + 
  ggtitle('Survival curve estimation') + 
  ylab('S(t)') + 
  xlab('t')

ggsave(filename = 'outputs_images/random_censoring/plot_manually_corrected.png', 
       scale=0.3, height = 275, width = 700, units = 'mm', dpi = 600)

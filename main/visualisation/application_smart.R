##------------------------------------------------##
##  Author:     
##  University: 
##  Title:      Visualisation for application of
##              KM and CKM estimates
##  Year:       2022
##  Run on:     Apple aarch64, R 4.1.2
##------------------------------------------------##

project_dir <- ''

rm(list=ls())
setwd(project_dir)
library(tidyverse)
library(survival)
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

#### Load data ####

data_cv <- as_tibble(hdnom::smart)
names(data_cv) <- names(data_cv) %>% tolower()

data_cv %>% 
  mutate(
    sex_male = if_else(sex == 1, 1, 0),
    
    smoker_yes = if_else(smoking == 3, 1, 0), 
    smoker_never = if_else(smoking == 1, 1, 0), 
    
    alcohol_yes = if_else(alcohol == 3, 1, 0), 
    alcohol_never = if_else(alcohol == 1, 1, 0), 
    
    albumin_micro = if_else(albumin == 2, 1, 0),
    albumin_macro = if_else(albumin == 3, 1, 0),
  ) -> data_prep

data_prep %>% 
  rowid_to_column('rowid')-> data_prep

data_prep %>% 
  mutate(tevent = cut(tevent,
                       seq(0,3500, 100), 
                       labels=(seq(100,3500, 100)/100))) %>% 
  mutate(tevent = as.numeric(tevent)) %>% 
  select(-c(smoking, alcohol, sex, albumin)) -> data_prep

set.seed(42)
data_prep %>% 
  sample_n(size = round(nrow(data_prep)*0.8)) -> train_

data_prep %>% 
  filter(!rowid %in% train_$rowid) -> test_

train_ %>% 
  write_csv('data/application/application_training_smart.csv')

test_ %>% 
  write_csv('data/application/application_test_smart.csv')

cox_train <- coxph(Surv(tevent, event) ~ ., data=train_ %>% select(-c(rowid)))
cox_train %>% summary()

model_resid <- cox.zph(cox_train)

#### Visualisation hazards ####

km_preds = read_csv('data/application/results_smart_km.csv')
cox_preds = read_csv('data/application/results_smart_cox.csv')
ckm_preds = read_csv('data/application/results_smart_ckm.csv')

test_ %>% 
  rowid_to_column('test_index') %>% 
  filter(event == 0) %>% 
  pull(test_index) -> censored_observations

test_ %>% 
  rowid_to_column('test_index') %>% 
  filter(event == 1) %>% 
  pull(test_index) -> real_observations

names(cox_preds) <- paste('obs_', names(cox_preds), sep='')
cox_censored = cox_preds[,censored_observations]
cox_uncensored = cox_preds[,real_observations]

names(ckm_preds) <- paste('obs_', names(ckm_preds), sep='')
ckm_censored = ckm_preds[,censored_observations]
ckm_uncensored = ckm_preds[,real_observations]

ckm_uncensored %>% 
  mutate(timeline = c(0:35)) %>% 
  mutate(type = 'event') %>% 
  mutate(estimator = 'CKM') %>% 
  gather(observation, survival_proba, c(obs_7:obs_688)) %>% 
  bind_rows(
    ckm_censored %>% 
      mutate(timeline = c(0:35)) %>% 
      mutate(type = 'censored') %>% 
      mutate(estimator = 'CKM') %>% 
      gather(observation, survival_proba, c(obs_0:obs_774))
  ) %>% 
  bind_rows(
    cox_censored %>% 
      mutate(timeline = c(1:35)) %>% 
      mutate(type = 'censored') %>% 
      mutate(estimator = 'Cox') %>% 
      gather(observation, survival_proba, c(obs_0:obs_774))
  ) %>%  
  bind_rows(
    cox_uncensored %>% 
      mutate(timeline = c(1:35)) %>% 
      mutate(type= 'event') %>% 
      mutate(estimator = 'Cox') %>% 
      gather(observation, survival_proba, c(obs_7:obs_688))
  ) %>% 
  bind_rows(
    km_preds %>% 
      mutate(type= 'censored', 
             estimator = 'KM', 
             observation = 'obs_1', 
             survival_proba = KM_estimate) %>% 
      select(-c(KM_estimate))
  ) -> data_predictions

data_predictions %>% 
  group_by(timeline, type, estimator) %>%
  summarise(
    survival = mean(survival_proba)
  ) %>% 
  ggplot() + 
  global_theme() + 
  scale_color_manual(values=c('gray', 'black', 'gray3'))  +
  scale_fill_manual(values = c('black', 'red', 'black')) +
  scale_linetype_manual(values = c('dashed', 'dotted', 'solid')) +
  geom_smooth(aes(x=timeline, y=survival,color=type,
                  linetype=estimator),
              se=F) + 
  ylab('S(t)') + 
  xlab('t') +
  guides(
    linetype = guide_legend(override.aes = list(color='black'))
  )

ggsave(filename = 'outputs_images/application_smart/hazards_between_mods.png', 
       scale=0.3, height = 275, width = 700, units = 'mm', dpi = 600)

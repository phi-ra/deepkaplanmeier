##------------------------------------------------##
##  Author:     
##  University: 
##  Title:      Visualisation for application of
##              KM and CKM estimates
##  Year:       2022
##  Run on:     Apple aarch64, R 4.1.2
##------------------------------------------------##
rm(list=ls())
project_dir <- ''

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

df_raw <- read_csv('https://raw.githubusercontent.com/ErikinBC/SurvSet/main/SurvSet/_datagen/output/UnempDur.csv')

df_raw %>% 
  rowid_to_column() -> df

df %>% 
  mutate(fac_ui = if_else(fac_ui == 'yes', 1, 0)) -> df

paste('Surv(time, event) ~', paste('num_age',
                                   'num_reprate', 
                                   'num_disrate', 
                                   'num_tenure',
                                   'num_logwage', 
                                   'fac_ui',
                                   sep=' + ')) %>% 
  as.formula() -> formula_rep

cox_mod_rep <- coxph(formula_rep, data=df)
summary(cox_mod_rep)

model_resid <- cox.zph(cox_mod_rep)
model_resid

model_resid$y[,'fac_ui'] %>% 
  names() %>% 
  as.numeric() -> time_steps_schoen

model_resid$y[,'fac_ui'] %>% 
  as_tibble() %>% 
  select(residual = value) %>% 
  add_column(time_step = time_steps_schoen) -> plot_resids

plot_resids %>% 
  ggplot() + 
  geom_point(aes(x=time_step, y=residual)) + 
  geom_smooth(aes(x=time_step, y=residual), 
              color='grey', se=F) + 
  global_theme() + 
  ylab('Beta(t) for UI dummy') + 
  xlab('t')

ggsave(filename = 'outputs_images/application_unemp/schoenfeld_resid.png', 
       scale=0.3, height = 275, width = 700, units = 'mm', dpi = 600)

#### Data split and preds ####
set.seed(42)
df %>% 
  sample_n(size = round(nrow(df)*0.8)) -> train_

df %>% 
  filter(!rowid %in% train_$rowid) -> test_

train_ %>% 
  write_csv('data/application/data_econ_train.csv')

test_ %>% 
  write_csv('data/application/data_econ_test.csv')

#### Average user example ####

tibble(
  num_age = rep(median(train_$num_age), 2), 
  num_reprate = rep(mean(train_$num_reprate), 2), 
  num_disrate = rep(mean(train_$num_disrate), 2), 
  num_tenure = rep(median(train_$num_tenure), 2), 
  num_logwage = rep(mean(train_$num_logwage), 2), 
  fac_ui = c('yes', 'no')) -> example_data

example_data %>% 
  write_csv('data/application/example_econ.csv')

## Visualize

data_example_prediction <- read_csv('data/application/econ_predictions_examples.csv')
data_example_prediction %>% 
  rowid_to_column() -> data_example_prediction

names(data_example_prediction) <- c('timepoint', 'received', 'not received')

data_example_prediction %>% 
  mutate(
    haz_1 = (lag(received) - received)/(lag(received)), 
    haz_2 = (lag(`not received`) - `not received`) / (lag(`not received`))
  ) %>% 
  mutate(
    haz_1 = if_else(haz_1 < 0.001, 0.001, haz_1), # Clip to avoid large outliers
    haz_2 = if_else(haz_1 < 0.001, 0.001, haz_2), # Clip to avoid large outliers
  ) %>% 
  mutate(
    ratio = haz_2 / haz_1
  ) %>% 
  {.->> data_hazard_ratios} %>% 
  ggplot() + 
  geom_point(aes(x=timepoint, y=ratio)) + 
  geom_smooth(aes(x=timepoint, y=ratio), 
              color='grey', lty=2, se=F) + 
  # geom_hline(yintercept = hazard_ratio_cox) +
  global_theme() + 
  ggtitle('Estimated hazard ratio over time', subtitle = 'not received/received') + 
  ylab('Hazard ratio') + 
  xlab('t')

ggsave(filename = 'outputs_images/application_unemp/hazard_rates.png', 
       scale=0.3, height = 275, width = 700, units = 'mm', dpi = 600)


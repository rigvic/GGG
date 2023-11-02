library(tidyverse)
library(tidymodels)




GGG_missing <- read.csv("~/Downloads/trainWithMissingValues.csv")
GGG_train <- read.csv("~/Desktop/GGG/train.csv")
GGG_test <- read.csv("~/Desktop/GGG/test.csv")

# Data Imputation
my_recipe <- recipe(type ~., data = GGG_missing) %>%
  step_impute_bag(bone_length, impute_with = imp_vars(has_soul, color, type), trees = 500) %>%
  step_impute_bag(rotting_flesh, impute_with = imp_vars(has_soul, color, type, bone_length), trees = 500) %>%
  step_impute_bag(hair_length, impute_with = imp_vars(has_soul, color, type, bone_length, rotting_flesh), trees = 500)

prep <- prep(my_recipe)
bake <- bake(prep, new_data = GGG_missing)

rmse_vec(as.numeric(GGG_train[is.na(GGG_missing)]), bake[is.na(GGG_missing)])

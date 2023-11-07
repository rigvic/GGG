library(tidyverse)
library(tidymodels)
library(vroom)
# library(poissonreg)
library(glmnet)
library(rpart)
library(ranger)
# library(stacks)
library(embed)
library(discrim)
library(kknn)
library(themis)
library(keras)


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

# RF
my_recipe <- recipe(type ~. , data = GGG_train) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(type)) %>%
  step_normalize(all_numeric_predictors())

prep <- prep(my_recipe)
baked_train <- bake(prep, new_data = GGG_train)
baked_test <- bake(prep, new_data = GGG_test)

rf_mod <- rand_forest(mtry = tune(),
                      min_n = tune(),
                      trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")


rf_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(rf_mod)

tuning_grid <- grid_regular(mtry(range=c(1,ncol(GGG_train)-1)),
                            min_n(),
                            levels = 5)

folds <- vfold_cv(GGG_train, v = 10, repeats = 1)

CV_results <- rf_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

bestTune <- CV_results %>%
  select_best("accuracy")

final_wf <-
  rf_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = GGG_train)

rf_predictions <- final_wf %>%
  predict(GGG_test, type = "class")

rf_predictions <- rf_predictions %>%
  bind_cols(., GGG_test) %>%
  select(id, .pred_class) %>%
  rename(type = .pred_class)

vroom_write(x=rf_predictions, file="rf_predictions.csv", delim=",")

# Neural Networks
my_recipe <- recipe(type ~. , data = GGG_train) %>%
  update_role(id, new_role = "id") %>%
  step_mutate_at(color, fn = factor) %>%
  step_dummy(color) %>%
  step_range(all_numeric_predictors(), min = 0, max = 1)
  
nn_mod <- mlp(hidden_units = tune(),
                epochs = 50,) %>%
  set_engine("nnet") %>%
  set_mode("classification")

nn_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nn_mod)

tuning_grid <- grid_regular(hidden_units(range = c(1, 50)),
                                         levels = 5)
folds <- vfold_cv(GGG_train, v = 5, repeats = 1)

tuned_nn <- nn_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

tuned_nn %>% collect_metrics() %>%
  filter(.metric=="accuracy") %>%
  ggplot(aes(x=hidden_units, y=mean)) + geom_line()

bestTune <- tuned_nn %>%
  select_best("accuracy")

final_wf <-
  nn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = GGG_train)

nn_predictions <- final_wf %>%
  predict(GGG_test, type = "class")

nn_predictions <- nn_predictions %>%
  bind_cols(., GGG_test) %>%
  select(id, .pred_class) %>%
  rename(type = .pred_class)

vroom_write(x=nn_predictions, file="nn_predictions.csv", delim=",")

# Naive Bayes
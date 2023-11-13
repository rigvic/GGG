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
my_recipe <- recipe(type ~., data=GGG_train) %>%
  step_lencode_glm(all_nominal_predictors(), outcome=vars(type)) %>%
  step_interact(~ hair_length + bone_length) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_range(all_numeric_predictors(), min = 0, max = 1)

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

# SVM
my_recipe_svm <- recipe(type ~., data=GGG_train) %>%
  step_lencode_glm(all_nominal_predictors(), outcome=vars(type)) %>%
  step_interact(~ hair_length + bone_length) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_range(all_numeric_predictors(), min = 0, max = 1)


## SVM Radial

svmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")

wf_svm <- workflow() %>%
  add_recipe(my_recipe_svm) %>%
  add_model(svmRadial)

## Fit or Tune Model
tuning_grid_svm <- grid_regular(rbf_sigma(),
                                cost(),
                                levels = 5) ## L^2 total tuning possibilities

## Set up K-fold CV
folds <- vfold_cv(GGG_train, v = 5, repeats=1)

CV_results_svm <- wf_svm %>%
  tune_grid(resamples=folds,
            grid=tuning_grid_svm,
            metrics=metric_set(accuracy))

## Find best tuning parameters
bestTune_svm <- CV_results_svm %>%
  select_best("accuracy")

final_wf_svm <- wf_svm %>%
  finalize_workflow(bestTune_svm) %>%
  fit(data=GGG_train)

## Predict
predictions_svm <- final_wf_svm %>%
  predict(GGG_test, type = "class")

predictions_svm <- predictions_svm %>%
  bind_cols(., GGG_test) %>%
  select(id, .pred_class) %>%
  rename(type = .pred_class)

vroom_write(x= predictions_svm, file="predictions_svm_radial.csv", delim=",")

# Boosting 
library(bonsai)
library(lightgbm)

my_recipe <- recipe(type ~. , data = GGG_train) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(type)) %>%
  step_normalize(all_numeric_predictors())

prep <- prep(my_recipe)
baked_train <- bake(prep, new_data = GGG_train)
baked_test <- bake(prep, new_data = GGG_test)

boost_mod <- boost_tree(tree_depth = tune(),
                        trees = tune(),
                        learn_rate = tune()) %>%
  set_engine("lightgbm") %>%
  set_mode("classification")


boost_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(boost_mod)

tuning_grid <- grid_regular(tree_depth(),
                            trees(),
                            learn_rate(),
                            levels = 5)

folds <- vfold_cv(GGG_train, v = 10, repeats = 1)

CV_results <- boost_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

bestTune <- CV_results %>%
  select_best("accuracy")

final_wf <-
  boost_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = GGG_train)

boost_predictions <- final_wf %>%
  predict(GGG_test, type = "class")

boost_predictions <- boost_predictions %>%
  bind_cols(., GGG_test) %>%
  select(id, .pred_class) %>%
  rename(type = .pred_class)

vroom_write(x=boost_predictions, file="boost_predictions.csv", delim=",")


# BART
my_recipe <- recipe(type ~. , data = GGG_train) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(type)) %>%
  step_normalize(all_numeric_predictors())

prep <- prep(my_recipe)
baked_train <- bake(prep, new_data = GGG_train)
baked_test <- bake(prep, new_data = GGG_test)

bart_mod <- parsnip::bart(trees = tune()) %>%
  set_engine("dbarts") %>%
  set_mode("classification")

bart_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(bart_mod)

tuning_grid <- grid_regular(trees(),
                            levels = 5)

folds <- vfold_cv(GGG_train, v = 10, repeats = 1)

CV_results <- bart_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

bestTune <- CV_results %>%
  select_best("accuracy")

final_wf <-
  bart_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = GGG_train)

bart_predictions <- final_wf %>%
  predict(GGG_test, type = "class")

bart_predictions <- bart_predictions %>%
  bind_cols(., GGG_test) %>%
  select(id, .pred_class) %>%
  rename(type = .pred_class)

vroom_write(x=bart_predictions, file="bart_predictions.csv", delim=",")

# Naive Bayes
my_recipe <- recipe(type ~., data=GGG_train) %>%
  step_lencode_glm(all_nominal_predictors(), outcome=vars(type)) %>%
  step_interact(~ hair_length + bone_length) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_range(all_numeric_predictors(), min = 0, max = 1)
  

nb_mod <- naive_Bayes(Laplace = tune(), smoothness = tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_mod)

tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 5)

folds <- vfold_cv(GGG_train, v = 5, repeats = 1)

CV_results <- nb_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

bestTune <- CV_results %>%
  select_best("accuracy")

final_wf <-
  nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = GGG_train)

nb_predictions <- final_wf %>%
  predict(GGG_test, type = "class")

nb_predictions <- nb_predictions %>%
  bind_cols(., GGG_test) %>%
  select(id, .pred_class) %>%
  rename(type = .pred_class)

vroom_write(x=nb_predictions, file="nb_predictions.csv", delim=",")

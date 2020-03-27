# Copied over from https://bitbucket.org/quantco/wayfairelastpricing/

rm(list = ls())
library(tidyverse)
library(glmnet)
setwd("C:\\Users\\esantorella\\code\\elasticity_estimation\\tests\\")

n_rows = 1000

df <- data.frame(y = pmax(.5, seq(0, n_rows - 1) / (n_rows - 1)),
                 x1 = rep(1, n_rows),
                 x2 = seq(0, n_rows - 1))

x = model.matrix(~x1 + x2 - 1, df)

get_glmnet_model_outputs <- function(model) {
  data.frame(
    lambda_ = model$lambda,
    beta_0 = model$beta[1,],
    beta_1 = model$beta[2,],
    dev_ratio = model$dev.ratio,
    df = model$df,
    npasses = model$npasses
  )
}


get_glmnet_basic_model <- function() {
  model <- glmnet(
    y              = cbind(1 - df$y, df$y),
    x              = x,
    family         = 'binomial',
    intercept      = FALSE
  )
  list(model=model, name='glmnet_base')
}

get_glmnet_varied_penalty_model <- function() {
  model <- glmnet(
    y              = cbind(1 - df$y, df$y),
    x              = x,
    family         = 'binomial',
    intercept      = FALSE,
    penalty.factor = c(0, 2, 1)
  )
  list(model=model, name='glmnet_varied_penalty')
}

get_glmnet_one_penalized_model <- function() {
  model <- glmnet(
    y              = cbind(1 - df$y, df$y),
    x              = x,
    family         = 'binomial',
    intercept      = FALSE,
    penalty.factor = c(0, 1)
  )
  list(model=model, name='glmnet_one_penalized')
}

get_glmnet_constrained_model <- function() {
  model <- glmnet(
    y              = cbind(1 - df$y, df$y),
    x              = x,
    family         = 'binomial',
    intercept      = FALSE,
    upper.limits = c(Inf, 1e-4)
  )
  list(model=model, name='glmnet_constrained')
}

get_cvglmnet_outputs <- function(model) {
  data.frame(
    lambda_ = model$lambda,
    beta_0 = model$glmnet.fit$beta[1,1:ncol(model$glmnet.fit$beta)],
    beta_1 = model$glmnet.fit$beta[2,1:ncol(model$glmnet.fit$beta)],
    cvm = model$cvm,
    cvsd = model$cvsd,
    nzero = model$nzero,
    best_lambda = model$lambda.min,
    best_beta_0 = coef(model, s = model$lambda.min)[2],
    best_beta_1 = coef(model, s = model$lambda.min)[3]
  )
}

get_cvglmnet_basic <- function() {
  model <- cv.glmnet(
    y              = cbind(1 - df$y, df$y),
    x              = x,
    family         = 'binomial',
    intercept      = FALSE
  )
  list(model = model, name = 'cvglmnet_basic')
}


get_cvglmnet_basic_foldid_specified <- function() {
  model <- cv.glmnet(
    y              = cbind(1 - df$y, df$y),
    x              = x,
    family         = 'binomial',
    intercept      = FALSE,
    foldid = seq(n_rows) %% 10 + 1
  )
  list(model = model, name = 'cvglmnet_foldid')
}

glmnet_dfs <- list(get_glmnet_basic_model(),
                   get_glmnet_varied_penalty_model(),
                   get_glmnet_one_penalized_model(),
                   get_glmnet_constrained_model()) %>% 
  lapply(function(x) get_glmnet_model_outputs(x$model) %>%
           mutate(name = x$name)) %>%
  bind_rows()

cvglmnet_dfs <- list(
                     get_cvglmnet_basic(),
                     get_cvglmnet_basic_foldid_specified()) %>%
  lapply(function(x) get_cvglmnet_outputs(x$model) %>%
           mutate(name = x$name)) %>%
  bind_rows()

write.csv(list(glmnet_dfs, cvglmnet_dfs) %>%
            bind_rows(), file = 'test_data/glmnet_results_from_R.csv')

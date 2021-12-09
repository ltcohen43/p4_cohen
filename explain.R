library(tidyverse)
library(tidymodels)
library(DALEX)
library(vip)


httpgd::hgd()
httpgd::hgd_browse()

dat_ml <- read_rds("dat_ml.rds") %>%
    select(arcstyle_ONE.AND.HALF.STORY, arcstyle_ONE.STORY, numbaths,
        tasp, livearea, basement, condition, stories, quality, before1980) %>%
    filter(livearea < 5500) # 99th percentile is 5429.04

set.seed(76)
dat_split <- initial_split(dat_ml, prop = 2/3, strata = before1980)

dat_train <- training(dat_split)
dat_test <- testing(dat_split)
# before is now 0 and after is 1
dat_exp <- mutate(dat_train, before1980 = as.integer(dat_train$before1980) - 1) #change 1980 factor to integer

head(dat_exp$before1980)
head(dat_train$before1980)

bt_model <- boost_tree() %>%
    set_engine(engine = "xgboost") %>%
    set_mode("classification") %>%
    fit(before1980 ~ ., data = dat_train)

logistic_model <- logistic_reg() %>%
    set_engine(engine = "glm") %>%
    set_mode("classification") %>%
    fit(before1980 ~ ., data = dat_train)

vip(bt_model)
vip(logistic_model)

explainer_bt <- DALEX::explain(
    bt_model,
    select(dat_exp, -before1980), dat_exp$before1980, label = "Boosted Trees")

explainer_logistic <- DALEX::explain(
    logistic_model,
    select(dat_exp, -before1980), dat_exp$before1980, label = "Logistic Regression") # nolint

performance_logistic <- model_performance(explainer_logistic)
performance_bt <- model_performance(explainer_bt)

plot(performance_bt, performance_logistic) 
#boosted tree is a better model, can see they have lower number of residuals higher number # nollint
plot(performance_bt, performance_logistic, geom = "boxplot")

logistic_parts <- model_parts(explainer_logistic,
    loss_function = loss_root_mean_square)
bt_parts <- model_parts(explainer_bt,
    loss_function = loss_root_mean_square)

plot(logistic_parts, bt_parts, max_vars = 10)


logistic_parts <- model_parts(explainer_logistic,
    loss_function = loss_root_mean_square, type = "difference")
bt_parts <- model_parts(explainer_bt,
    loss_function = loss_root_mean_square)
plot(logistic_parts, bt_parts, max_vars = 10)

library(patchwork)

onehouse_before <- predict_parts(explainer_bt,
    new_observation = select(dat_exp, -before1980) %>%
        dplyr::slice(13800), type = "break_down")

onehouse_after <- predict_parts(explainer_bt,
    new_observation = select(dat_exp, -before1980) %>%
        dplyr::slice(8), type = "break_down")

plot(onehouse_after) + plot(onehouse_before)  #get to different structures each model
dat_train %>% dplyr::slice(c(8, 13800))


onehouse_before <- predict_parts(explainer_bt,
    new_observation = select(dat_exp, -before1980) %>%
        dplyr::slice(13800), type = "shap")

onehouse_after <- predict_parts(explainer_bt,
    new_observation = select(dat_exp, -before1980) %>%
        dplyr::slice(8), type = "shap")

plot(onehouse_after) + plot(onehouse_before)
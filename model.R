library(tidyverse)
library(tidymodels)
library(DALEX)
library(skimr)
library(GGally)
library(xgboost)
library(vip)
library(patchwork)

httpgd::hgd()
httpgd::hgd_browse()

dat_ml <- read_rds("dat_ml.rds")

set.seed(76) # run seed each time to get smme numbers
dat_split <- initial_split(dat_ml, prop = 2/3, strata = before1980) # proportion set to on test data, balance on 1980 # nolint

dat_train <- training(dat_split)
dat_test <- testing(dat_split)

bt_model <- boost_tree() %>%
    set_engine(engine = "xgboost") %>%
    set_mode("classification") %>%
    fit(before1980 ~ ., data = dat_train)

logistic_model <- logistic_reg() %>%
    set_engine(engine = "glm") %>%
    set_mode("classification") %>%
    fit(before1980 ~ ., data = dat_train)

nb_model <- discrim::naive_Bayes() %>%
    set_engine(engine = "naivebayes") %>%
    set_mode("classification") %>%
    fit(before1980 ~ ., data = dat_train)

(vip(bt_model, num_features = 20) + labs(title = "boosted")) +
(vip(logistic_model, num_features = 20) + labs(title = "logistic"))

preds_logistic <- bind_cols(
    predict(logistic_model, new_data = dat_test),
    predict(logistic_model, dat_test, type = "prob"),
    truth = pull(dat_test, before1980)
  )


preds_logistic <- bind_cols(
    predict(logistic_model, new_data = dat_test),
    predict(logistic_model, dat_test, type = "prob"),
    truth = pull(dat_test, before1980)
  )

preds_nb <- bind_cols(
    predict(nb_model, new_data = dat_test),
    predict(nb_model, dat_test, type = "prob"),
    truth = pull(dat_test, before1980)
  )

preds_bt <- bind_cols(
    predict(bt_model, new_data = dat_test),
    predict(bt_model, dat_test, type = "prob"),
    truth = pull(dat_test, before1980)
  )

preds_bt %>% conf_mat(truth, .pred_class)
preds_nb %>% conf_mat(truth, .pred_class)
preds_logistic %>% conf_mat(truth, .pred_class)

preds_bt %>% metrics(truth, .pred_class)

metrics_calc <- metric_set(accuracy, bal_accuracy, precision, recall, f_meas)

preds_bt %>%
    metrics_calc(truth, estimate = .pred_class)

preds_nb %>%
    metrics_calc(truth, estimate = .pred_class)

preds_bt %>%
    roc_curve(truth, estimate = .pred_before) %>%
    autoplot()

preds_nb %>%
    roc_curve(truth, estimate = .pred_before) %>%
    autoplot()


preds_all <- bind_rows(
    mutate(preds_nb, model = "Naive Bayes"),
    mutate(preds_bt, model = "Boosted Tree"),
    mutate(preds_logistic, model = "Logistic Regression")
)

preds_all %>%
    group_by(model) %>%
    roc_curve(truth, estimate = .pred_before) %>%
    autoplot()

preds_all %>%
    group_by(model) %>%
    metrics_calc(truth, estimate = .pred_class) %>%
    pivot_wider(names_from = .metric, values_from = .estimate)


# # takes a minute
# preds_nb <- bind_cols(
#     predict(nb_model, new_data = dat_test),
#     predict(nb_model, dat_test, type = "prob"),
#     truth = pull(dat_test, before1980)
#   )

# preds_bt <- bind_cols(
#     predict(bt_model, new_data = dat_test),
#     predict(bt_model, dat_test, type = "prob"),
#     truth = pull(dat_test, before1980)
#   )

# preds_bt %>% conf_mat(truth, .pred_class)
# preds_nb %>% conf_mat(truth, .pred_class)
# preds_logistic %>% conf_mat(truth, .pred_class)

# metrics_calc <- metric_set(accuracy, bal_accuracy, precision, recall, f_meas)

# preds_bt %>% metrics_calc(truth, estimate = .pred_class)

# preds_nb %>% metrics_calc(truth, estimate = .pred_class)

# preds_bt %>% roc_curve(truth, estimate = .pred_before) %>% autoplot()

# preds_nb %>% roc_curve(truth, estimate = .pred_before) %>% autoplot()

# preds_logistic %>% roc_curve(truth, estimate = .pred_before) %>% autoplot()

# preds_all <- bind_rows(
#     mutate(preds_nb, model = "Naive Bayes"),
#     mutate(preds_bt, model = "Boosted Tree"),
#     mutate(preds_logistic, model = "Logistic Regression")
# )

# #preds_all %>% roc_curve(truth, estimate = .pred_before) %>% autoplot()

# conf_mat()
# metrics()
# precision()
# metric_set()
# roc_curve()
# autoplot()
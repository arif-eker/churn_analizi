from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

rf_params = {"max_depth": [3, 5, 8],
             "max_features": [8, 15, 25],
             "n_estimators": [200, 500, 1000],
             "min_samples_split": [2, 5, 10]}

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [200, 500, 1000],
               "max_depth": [3, 5, 8],
               "colsample_bytree": [1, 0.8, 0.5],
               "num_leaves": [32, 64, 128]}

xgb_params = {"learning_rate": [0.1, 0.01],
              "max_depth": [3, 5, 8],
              "n_estimators": [200, 500, 1000],
              "colsample_bytree": [0.7, 1]}


def get_tuned_models(x_train, y_train, rnd_state):
    lgb_model = LGBMClassifier(random_state=rnd_state)
    rf_model = RandomForestClassifier(random_state=rnd_state)
    xgb_model = XGBClassifier(random_state=rnd_state)

    lgbm_cv_model = GridSearchCV(lgb_model,
                                 lgbm_params,
                                 cv=10,
                                 n_jobs=-1,
                                 verbose=2).fit(x_train, y_train)

    rf_cv_model = GridSearchCV(rf_model,
                               rf_params,
                               cv=10,
                               n_jobs=-1,
                               verbose=2).fit(x_train, y_train)

    xgb_cv_model = GridSearchCV(xgb_model,
                                xgb_params,
                                cv=10,
                                n_jobs=-1,
                                verbose=2).fit(x_train, y_train)

    lgbm_tuned = LGBMClassifier(**lgbm_cv_model.best_params_, random_state=rnd_state).fit(x_train, y_train)

    rf_tuned = RandomForestClassifier(**rf_cv_model.best_params_, random_state=rnd_state).fit(x_train, y_train)

    xgb_tuned = XGBClassifier(**xgb_cv_model.best_params_, random_state=rnd_state).fit(x_train, y_train)

    return lgbm_tuned, rf_tuned, xgb_tuned

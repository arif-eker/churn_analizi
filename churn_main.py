#
#

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import scripts.helpers as hlp
from scipy.stats import shapiro
import scripts.models as models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/churn.csv")

# Verinin baştan ve sondan ilk 5 gözlemi
df.head()
df.tail()

# Kaç farklı müşteri var?
df["CustomerId"].nunique()

# Anlamsız değişkenlerin düşürülmesi
need_drops = ["RowNumber", "CustomerId", "Surname"]
df.drop(need_drops, axis=1, inplace=True)

# Eksik gözlem kontrolü
df.isnull().sum()

# Betimsel istatistiklere bakalım
df.describe([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).T

# Kategorik değişkenlerin hedefle ilişkisinin incelenmesi
temp_categorical = ["Geography", "Gender", "Tenure", "NumOfProducts", "HasCrCard", "IsActiveMember"]
hlp.cat_summary(df, temp_categorical, "Exited")

# Sayısal değişkenler için histogram incelenmesi
temp_numeric = ["CreditScore", "Age", "Balance", "EstimatedSalary"]
hlp.hist_for_numeric_columns(df, temp_numeric)

# Sayısal değişkenlerin normallik varsayımları
# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:... sağlanmamaktadır.

# p - value < ise 0.05'ten HO RED.
# p - value < değilse 0.05 H0 REDDEDİLEMEZ.
creditscore = df[["CreditScore"]]
balance = df[["Balance"]]
salary = df[["EstimatedSalary"]]
ages = df[["Age"]]

shapiro(creditscore["CreditScore"])[1] < 0.05
shapiro(balance["Balance"])[1] < 0.05
shapiro(salary["EstimatedSalary"])[1] < 0.05
shapiro(ages["Age"])[1] < 0.05

# Bütün H0'lar reddedildi


# Korelasyonlara bakalım
low_corr_list, up_corr_list = hlp.find_correlation(df, temp_numeric, "Exited")

for i in low_corr_list:
    print(i)

for i in up_corr_list:
    print(i)

# Feature Engineering
hlp.add_features(df)

categorical_columns = ["Geography", "Gender", "NumOfProducts",
                       "NEW_Age_Range", "NEW_Tenure_Status",
                       "NEW_CreditScore_Status", "NEW_EstimatedSalary_Status",
                       "NEW_MemberStarts_Age_Range"]

numerical_columns = ["CreditScore", "Age", "Tenure", "Balance",
                     "EstimatedSalary",
                     "NEW_Card_Member_Score", "NEW_MemberStarts_Age"]

hlp.rare_analyser(df, categorical_columns, "Exited", 0.5)

# Nadir sınıflar silinecek
df = df.loc[~((df["NumOfProducts"] == 3) | (df["NumOfProducts"] == 4))]

# One-Hot Encode
df, one_hot_columns = hlp.one_hot_encoder(df, categorical_columns)

# Robust Scale
need_scale_cols = ["Balance", "EstimatedSalary"]
for col in need_scale_cols:
    df[col] = hlp.robust_scaler(df[col])

df.describe([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).T

X = df.drop("Exited", axis=1)
y = np.ravel(df[["Exited"]])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=123)

lgbm_tuned, rf_tuned, xgb_tuned = models.get_tuned_models(X_train, y_train, 123)

# Accuracy results
models = [("RF", rf_tuned),
          ("LGBM", lgbm_tuned),
          ("XGB", xgb_tuned)]

for name, model in models:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    msg = "%s: (%f)" % (name, acc)
    print(msg)

# ************************************************************************* #

# Modelleri kaydetme
tuned_models = [("RF", rf_tuned),
                ("LGBM", lgbm_tuned),
                ("XGB", xgb_tuned)]

for model in tuned_models:
    pickle.dump(model[1], open("nonlinear_models/" + str(model[0]) + ".pkl", "wb"))

# df.to_pickle("data/churn_analysis_pure.pkl")

# Kaydedilen df'ten modeli kurma

pure_df = pd.read_pickle("data/churn_analysis_pure.pkl")

X_pure = pure_df.drop("Exited", axis=1)
y_pure = np.ravel(pure_df[["Exited"]])

X_pure_train, X_pure_test, y_pure_train, y_pure_test = train_test_split(X_pure,
                                                                        y_pure,
                                                                        test_size=0.35,
                                                                        random_state=123)

# Kaydedilen modeli çağırma
rf_from_pickle = pickle.load(open("nonlinear_models/RF.pkl", "rb"))

lgbm_from_pickle = pickle.load(open("nonlinear_models/LGBM.pkl", "rb"))

xgb_from_pickle = pickle.load(open("nonlinear_models/XGB.pkl", "rb"))

model_from_pickle = [("RF", rf_from_pickle),
                     ("LGBM", lgbm_from_pickle),
                     ("XGB", lgbm_from_pickle)]

result = []
for name, model in model_from_pickle:
    y_pred = model.predict(X_pure_test)
    acc = accuracy_score(y_pure_test, y_pred)
    msg = "%s: (%f)" % (name, acc)
    result.append(msg)
    print(msg)

print("******************** CHURN PREDICTION ********************")
for i in result:
    print(i)
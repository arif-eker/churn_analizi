import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn import preprocessing


low_q1 = 0.05
upper_q3 = 0.95
correlation_limit = 0.60

def cat_summary(dataframe, categorical_columns, target, plot=False):
    """
    -> Kategorik değişkenlerin sınıflarının oranını ve targettaki medyanı gösterir.

    :param dataframe: İşlem yapılacak dataframe
    :param categorical_columns: Kategorik değişkenlerin adları
    :param target: Dataframe'de ilgilendiğimiz değişken.
    :param plot: Grafik çizdirmek için argüman : True/False

    """
    for col in categorical_columns:
        print(col, " : ", dataframe[col].nunique(), " unique classes.\n")

        print(col, " : ", dataframe[col].value_counts().sum(), "\n")

        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO ( % )": 100 * dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEDIAN": dataframe.groupby(col)[target].median(),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

        if plot:
            sns.countplot(x=col, data=dataframe)

            plt.show()


def hist_for_numeric_columns(dataframe, numeric_columns):
    """
    -> Sayısal değişkenlerin histogramını çizdirir.

    :param dataframe: İşlem yapılacak dataframe.
    :param numeric_columns: Sayısal değişkenlerin adları

    """
    col_counter = 0

    data = dataframe.copy()

    for col in numeric_columns:
        data[col].hist(bins=20)

        plt.xlabel(col)

        plt.title(col)

        plt.show()

        col_counter += 1

    print(col_counter, "variables have been plotted!")


def find_correlation(dataframe, numeric_columns, target, corr_limit=correlation_limit):
    """
    -> Sayısal değişkenlerin targetla olan korelasyonunu inceler.

    :param dataframe: İşlem yapılacak dataframe
    :param numeric_columns: Sayısal değişken adları
    :param target: Korelasyon ilişkisinde bakılacak hedef değişken
    :param corr_limit: Korelasyon sınırı. Sınırdan aşağısı düşük, yukarısı yüksek korelasyon
    :return: İlk değer düşük korelasyona sahip değişkenler, ikinci değer yüksek korelasyona sahip değişkenler
    """
    high_correlations = []

    low_correlations = []

    for col in numeric_columns:
        if col == target:
            pass

        else:
            correlation = dataframe[[col, target]].corr().loc[col, target]

            if abs(correlation) > corr_limit:
                high_correlations.append(col + " : " + str(correlation))

            else:
                low_correlations.append(col + " : " + str(correlation))

    return low_correlations, high_correlations


def outlier_thresholds(dataframe, variable, low_quantile=low_q1, up_quantile=upper_q3):
    """
    -> Verilen değerin alt ve üst aykırı değerlerini hesaplar ve döndürür.

    :param dataframe: İşlem yapılacak dataframe
    :param variable: Aykırı değeri yakalanacak değişkenin adı
    :param low_quantile: Alt eşik değerin hesaplanması için bakılan quantile değeri
    :param up_quantile: Üst eşik değerin hesaplanması için bakılan quantile değeri
    :return: İlk değer olarak verilen değişkenin alt sınır değerini, ikinci değer olarak üst sınır değerini döndürür
    """
    quantile_one = dataframe[variable].quantile(low_quantile)

    quantile_three = dataframe[variable].quantile(up_quantile)

    interquantile_range = quantile_three - quantile_one

    up_limit = quantile_three + 1.5 * interquantile_range

    low_limit = quantile_one - 1.5 * interquantile_range

    return low_limit, up_limit


def has_outliers(dataframe, numeric_columns, plot=False):
    """
    -> Sayısal değişkenlerde aykırı gözlem var mı?

    -> Varsa isteğe göre box plot çizdirme görevini yapar.

    -> Ayrıca aykırı gözleme sahip değişkenlerin ismini göndürür.

    :param dataframe:  İşlem yapılacak dataframe
    :param numeric_columns: Aykırı değerleri bakılacak sayısal değişken adları
    :param plot: Boxplot grafiğini çizdirmek için bool değer alır. True/False
    :return: Aykırı değerlere sahip değişkenlerin adlarını döner
    """
    variable_names = []

    for col in numeric_columns:
        low_limit, up_limit = outlier_thresholds(dataframe, col)

        if dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].any(axis=None):
            number_of_outliers = dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].shape[0]

            print(col, " : ", number_of_outliers, " aykırı gözlem.")

            variable_names.append(col)

            if plot:
                sns.boxplot(x=dataframe[col])
                plt.show()

    return variable_names


def replace_with_thresholds(dataframe, numeric_columns):
    """
    Baskılama yöntemi

    Silmemenin en iyi alternatifidir.

    Loc kullanıldığından dataframe içinde işlemi uygular.

    :param dataframe: İşlem yapılacak dataframe
    :param numeric_columns: Aykırı değerleri baskılanacak sayısal değişkenlerin adları
    """
    for variable in numeric_columns:
        low_limit, up_limit = outlier_thresholds(dataframe, variable)

        dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit

        dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def missing_values_table(dataframe):
    """
     Eksik değerlere sahip değişkenleri gösterir ve bu değerleri döndürür.

    :param dataframe: İşlem yapılacak dataframe
    :return: Eksik değerlere sahip değişkenlerin adlarını döndürür.
    """
    variables_with_na = [col for col in dataframe.columns
                         if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[variables_with_na].isnull().sum().sort_values(ascending=False)

    ratio = (dataframe[variables_with_na].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    # ratio = (100 * dataframe[variables_with_na].isnull().sum() / dataframe.shape[0]).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])

    print(missing_df)

    return variables_with_na


def missing_vs_target(dataframe, target, variable_with_na):
    """
    Bu fonksiyon, eksik değerlere sahip değişkenlerin target açısından etkisine bakmamızı sağlar.
    Yeni bir değişken oluşturur : incelenen değer + _NA_FLAG
    Bu yeni değişkene, incelenen değişkende eksik gördüğünde 1, eksik yoksa 0 değerlerini atar.
    Daha sonra bu değişkenlere göre gruplama yapıp, target incelenir.

    :param dataframe: İşlem yapılacak dataframe
    :param target: Analizi yapılacak hedef değişkenin adı
    :param variable_with_na: Eksik değerlere sahip değişkenlerin adı.

    """
    temp_df = dataframe.copy()

    for variable in variable_with_na:
        temp_df[variable + "_NA_FLAG"] = np.where(temp_df[variable].isnull(), 1, 0)

    flags_na = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for variable in flags_na:
        print(pd.DataFrame({"TARGET_MEDIAN": temp_df.groupby(variable)[target].median()}),
              end="\n\n\n")


def label_encoder(dataframe, categorical_columns):
    """
    2 sınıflı kategorik değişkeni 0-1 yapma

    :param dataframe: İşlem yapılacak dataframe
    :param categorical_columns: Label encode yapılacak kategorik değişken adları
    :return:
    """
    labelencoder = preprocessing.LabelEncoder()

    for col in categorical_columns:

        if dataframe[col].nunique() == 2:
            dataframe[col] = labelencoder.fit_transform(dataframe[col])

    return dataframe


def one_hot_encoder(dataframe, categorical_columns, nan_as_category=False):
    """
    Drop_first doğrusal modellerde yapılması gerekli

    Ağaç modellerde gerekli değil ama yapılabilir.

    dummy_na eksik değerlerden değişken türettirir.

    :param dataframe: İşlem yapılacak dataframe
    :param categorical_columns: One-Hot Encode uygulanacak kategorik değişken adları
    :param nan_as_category: NaN değişken oluştursun mu? True/False
    :return: One-Hot Encode yapılmış dataframe ve bu işlem sonrası oluşan yeni değişken adlarını döndürür.
    """
    original_columns = list(dataframe.columns)

    dataframe = pd.get_dummies(dataframe, columns=categorical_columns,
                               dummy_na=nan_as_category, drop_first=False)

    new_columns = [col for col in dataframe.columns if col not in original_columns]

    return dataframe, new_columns


def rare_analyser(dataframe, categorical_columns, target, rare_perc):
    """
     Data frame değişkenlerinin herhangi bir sınıfı, verilen eşik değerden düşük frekansa sahipse bu değişkenleri gösterir.

    :param dataframe: İşlem yapılacak dataframe
    :param categorical_columns: Rare analizi yapılacak kategorik değişken adları
    :param target: Analizi yapılacak hedef değişken adı
    :param rare_perc: Rare için sınır değer. Altında olanlar rare kategorisine girer.
    :return:
    """
    rare_columns = [col for col in categorical_columns
                    if (dataframe[col].value_counts() / len(dataframe) < rare_perc).any(axis=None)]

    for var in rare_columns:
        print(var, " : ", len(dataframe[var].value_counts()))

        print(pd.DataFrame({"COUNT": dataframe[var].value_counts(),
                            "RATIO": dataframe[var].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(var)[target].mean(),
                            "TARGET_MEDIAN": dataframe.groupby(var)[target].median()}),
              end="\n\n\n")

    print(len(rare_columns), " adet rare sınıfa sahip değişken var.")


def robust_scaler(variable):
    var_median = variable.median()
    quartile1 = variable.quantile(0.01)
    quartile3 = variable.quantile(0.99)
    interquantile_range = quartile3 - quartile1
    if int(interquantile_range) == 0:
        quartile1 = variable.quantile(0.05)
        quartile3 = variable.quantile(0.95)
        interquantile_range = quartile3 - quartile1
        if int(interquantile_range) == 0:
            quartile1 = variable.quantile(0.25)
            quartile3 = variable.quantile(0.75)
            interquantile_range = quartile3 - quartile1
            z = (variable - var_median) / interquantile_range
            return round(z, 3)

        z = (variable - var_median) / interquantile_range
        return round(z, 3)
    else:
        z = (variable - var_median) / interquantile_range
    return round(z, 3)

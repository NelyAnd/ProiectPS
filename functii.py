import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns

def analiza_valorilor_lipsa(dataframe):
    """
    Funcție pentru calulul valorilor lipsa, atât în valoare absolută, cât și procentajul raportat la nr. de înregistrări.

    Parametri:
    ----------
    dataframe : pd.DataFrame
        DataFrame-ul care conține datele analizate.

    Return:
    -------
    pd.DataFrame
        Tabel care conține următoarele coloane:
        - 'Coloană': Numele coloanelor din DataFrame.
        - 'Număr valori lipsă': Numărul de valori lipsă pentru fiecare coloană.
        - 'Procentaj (%)': Procentul valorilor lipsă pentru fiecare coloană.
        Rezultatul este sortat descrescător după numărul de valori lipsă.
    """

    nan_summary = dataframe.isnull().sum()
    nan_percentage = (dataframe.isnull().sum() / len(dataframe)) * 100
    return pd.DataFrame({
        'Coloană': nan_summary.index,
        'Număr valori lipsă': nan_summary.values,
        'Procentaj (%)': nan_percentage.values
    }).sort_values(by="Număr valori lipsă", ascending=False)

def plot_bar_valori_lipsa(valori_lipsa_df):
    """
    Creează un grafic de tip bară orizontală pentru vizualizarea distribuției valorilor lipsă.

    Parametri:
    ----------
    valori_lipsa_df : pd.DataFrame
        DataFrame-ul rezultat din analiza valorilor lipsă, care trebuie să conțină coloanele: coloană și procentaj

    Return:
    -------
    plt : matplotlib.pyplot
    """

    plt.figure(figsize=(10, 6))
    plt.barh(valori_lipsa_df["Coloană"], valori_lipsa_df["Procentaj (%)"], color='darkred')
    plt.xlabel("Procentaj (%)")
    plt.title("Distribuția valorilor lipsă")
    plt.tight_layout()
    plt.gca().invert_yaxis()
    return plt

def histograme_variabile(dataframe):
     """
    Creează histograme pentru variabilele numerice dintr-un DataFrame, excluzând coloana "YEAR".

    Parametri:
    ----------
    dataframe : pd.DataFrame
        DataFrame-ul care conține coloanele numerice pentru care se vor crea histograme.

    Return:
    -------
    plt : matplotlib.pyplot
        Obiectul matplotlib ce conține figura cu histogramele generate.
    """

    #Pentru noi toate datele sunt numerice momentan deci afisam toate coloanele numerice, in afara de YEAR
     numerical_cols = [col for col in dataframe.select_dtypes(include=[np.number]).columns if col.upper() != "YEAR"]

     n_cols = 4  # 4 grafice pe rând
     n_rows = math.ceil(len(numerical_cols) / n_cols)  #Număr de rânduri necesare

     plt.figure(figsize=(6 * n_cols, 4 * n_rows))

     for i, col in enumerate(numerical_cols, 1):
         plt.subplot(n_rows, n_cols, i)
         plt.hist(dataframe[col], bins=30, color='skyblue', edgecolor='black')
         plt.title(f'Histogramă {col}')
         plt.xlabel(col)
         plt.ylabel('Frecvență')

     plt.tight_layout()  #Ajustează spațierea între grafice
     return plt

def gaseste_outlieri_iqr(dataframe):
    """
   Calculează limitele inferioare și superioare folosind metoda iqr pentru fiecare coloană numerică
   (excluzând coloana "YEAR" dacă aceasta există) și returnează un dicționar cu rezultatele.

   Parametri:
   ----------
   dataframe : pandas.DataFrame
       DataFrame-ul care conține datele pentru analiză.

   Return:
   -------
   outliers_dict : dict
       Dicționar în care cheia este numele coloanei, iar valoarea este un tuplu:
       (lower_bound, upper_bound, outliers_df) pentru coloana respectivă.
   """
    outliers_dict = {}
    # Selectăm coloanele numerice, excluzând "YEAR"
    numerical_cols = [col for col in dataframe.select_dtypes(include=[np.number]).columns if col.upper() != "YEAR"]

    for col in numerical_cols:
        q1 = dataframe[col].quantile(0.25)  # Calculăm Quartila 1
        q3 = dataframe[col].quantile(0.75)  # Calculăm Quartila 3
        iqr = q3 - q1  # Intervalul intercuartilic
        lower_bound = q1 - 1.5 * iqr  # Limita inferioară
        upper_bound = q3 + 1.5 * iqr  # Limita superioară
        # Selectăm valorile care ies din intervalul [lower_bound, upper_bound]
        outliers_df = dataframe[(dataframe[col] < lower_bound) | (dataframe[col] > upper_bound)]
        outliers_dict[col] = (lower_bound, upper_bound, outliers_df)

    return outliers_dict

def tratare_outlieri(dataframe):
    """
    Tratează outlierii din DataFrame conform strategiilor specifice fiecărei coloane:

    - health_exp: Winsorization (înlocuirea valorilor peste limita superioară și sub limita inferioară cu valorile limită)
    - life_expect: Înlocuirea valorilor mai mici decât limita inferioară (conform metodei iqr) cu limita inferioară
    - maternal_mortality: Transformare logaritmică (log(x + 1)) pentru a reduce asimetria
    - infant_mortality: Trunchierea valorilor peste percentila 99 (top 1%)
    - neonatal_mortality: Nu se aplică tratament
    - under_5_mortality: Trunchierea valorilor peste percentila 99
    - inci_tuberc: Winsorization la percentila 99
    - prev_undernourishment: Winsorization la percentila 99

    Parametri:
    ----------
    dataframe : pd.DataFrame
        DataFrame-ul original cu datele.

    Return:
    -------
    df_tratare_outlieri : pd.DataFrame
        DataFrame-ul modificat după tratarea outlierilor.
    """
    df_tratare_outlieri = dataframe.copy()

    # 1. health_exp: Winsorization (înlocuirea valorilor extreme cu limitele calculate prin iqr)
    q1 = df_tratare_outlieri['health_exp'].quantile(0.25)
    q3 = df_tratare_outlieri['health_exp'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df_tratare_outlieri['health_exp'] = df_tratare_outlieri['health_exp'].apply(
        lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x)
    )

    # 2. life_expect: Înlocuirea valorilor mai mici decât limita inferioară cu limita inferioară
    q1 = df_tratare_outlieri['life_expect'].quantile(0.25)
    q3 = df_tratare_outlieri['life_expect'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    df_tratare_outlieri['life_expect'] = df_tratare_outlieri['life_expect'].apply(
        lambda x: lower_bound if x < lower_bound else x
    )

    # 3. maternal_mortality: Transformare logaritmică (log(x + 1))
    df_tratare_outlieri['maternal_mortality'] = df_tratare_outlieri['maternal_mortality'].apply(lambda x: np.log1p(x))

    # 4. infant_mortality: Trunchierea valorilor peste percentila 99
    infant_99 = df_tratare_outlieri['infant_mortality'].quantile(0.99)
    df_tratare_outlieri['infant_mortality'] = df_tratare_outlieri['infant_mortality'].apply(
        lambda x: infant_99 if x > infant_99 else x
    )

    # 5. neonatal_mortality: Nu se aplică tratament (rămâne neschimbată)

    # 6. under_5_mortality: Trunchierea valorilor peste percentila 95
    under5_95 = df_tratare_outlieri['under_5_mortality'].quantile(0.95)
    df_tratare_outlieri['under_5_mortality'] = df_tratare_outlieri['under_5_mortality'].apply(
        lambda x: under5_95 if x > under5_95 else x
    )

    # 7. inci_tuberc: Winsorization pe baza percentilei 90
    tuberc_90 = df_tratare_outlieri['inci_tuberc'].quantile(0.90)
    df_tratare_outlieri['inci_tuberc'] = df_tratare_outlieri['inci_tuberc'].apply(
        lambda x: tuberc_90 if x > tuberc_90 else x
    )

    # 8. prev_undernourishment: Winsorization pe baza percentilei 95
    undernour_95 = df_tratare_outlieri['prev_undernourishment'].quantile(0.95)
    df_tratare_outlieri['prev_undernourishment'] = df_tratare_outlieri['prev_undernourishment'].apply(
        lambda x: undernour_95 if x > undernour_95 else x
    )

    return df_tratare_outlieri

def afiseaza_matrice_corelatie(df, exclude_col='YEAR'):
    """
    Generează și returnează un heatmap cu matricea de corelație pentru variabilele numerice dintr-un DataFrame.

    Parametri:
    ----------
    df : pd.DataFrame
        Setul de date pe care se aplică analiza.
    exclude_col : str
        Numele unei coloane de exclus (de ex. 'YEAR') din analiza de corelație.

    Return:
    -------
    plt.Figure
        Figura matplotlib care conține heatmap-ul.
    """
    # Selectăm doar coloanele numerice, excludem coloana specificată
    numerical_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col.upper() != exclude_col.upper()]
    corr_matrix = df[numerical_cols].corr()

    # Construim heatmap-ul
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Matricea de corelație pentru variabilele numerice")
    plt.tight_layout()

    return fig

import streamlit as st
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from functii import *
st.title("Proiect pachete software")

st.sidebar.title("Navigație")
section = st.sidebar.radio("Alege categoria:", ["📊 Date preliminare","Încărcare și afișare", "Selecție și filtrare", "Vizualizare grafice",
                                                "🛠️ Prelucrarea datelor","Analiza valorilor","Tratarea valorilor lipsă","Prelucrarea avansată a datelor"])

if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.file_name = None
    st.session_state.df_prelucrat=None

if section == "📊 Date preliminare":
    st.header("📊 Date preliminare")
    st.subheader("Prezentarea generală a setului de date 📝")
    st.text("Setul de date include indicatori cheie de sănătate pentru 189 de țări, acoperind perioada 2010-2020.")
    st.subheader("Indicatori de sănătate 🏥")
    st.markdown("""
    - *Cheltuieli curente de sănătate (% din PIB)* – arată procentul din PIB-ul unei țări alocat cheltuielilor cu sănătatea.
    - *Speranța de viață la naștere (total de ani)* – numărul mediu de ani pe care se așteaptă să trăiască un nou-născut, presupunând că ratele mortalității specifice vârstei rămân constante.
    - *Mortalitatea maternă* – numărul de decese materne la 100.000 de născuți vii.
    - *Rata mortalității infantile* – numărul de decese infantile (sub 1 an) la 1.000 de născuți vii.
    - *Rata mortalității neonatale* – numărul deceselor copiilor sub 28 de zile la 1.000 de născuți vii.
    - *Rata mortalității sub 5 ani* – numărul deceselor copiilor sub 5 ani la 1.000 de născuți vii.
    - *Prevalența HIV (% din populație)* – procentul populației cu vârsta cuprinsă între 15 și 49 de ani care trăiește cu HIV.
    - *Incidența tuberculozei (la 100.000 de persoane)* – numărul de cazuri de tuberculoză la 100.000 de persoane.
    - *Prevalența subnutriției (% din populație)* – procentul din populație al cărui aport caloric este sub minimul necesar pentru o viață sănătoasă.
    """)
    st.subheader("Sursa datelor 🌐")
    st.text("Acest set de date este compilat din baza de date de sănătate a Băncii Mondiale, oferind statistici fiabile și actualizate privind indicatorii de sănătate la nivel mondial.")
    st.subheader("Coloanele din setul de date:")
    st.markdown("""
     - *country* :  numele tarii pentru care sunt raportate datele.
     - *country_code* : codul de țară ISO.
     - *year* : anul pentru care sunt raportate datele.
     - *health_exp* : cheltuielile curente pentru sănătate ca procent din PIB.
     - *life_expect* : speranța de viață la naștere (total de ani) pentru populație.
     - *maternal_mortality* : numărul deceselor materne la 100.000 de născuți vii.
     - *infant_mortality*  : numărul de decese infantile (sub 1 an) la 1.000 de născuți vii.
     - *neonatal_mortality* : numărul deceselor copiilor sub 28 de zile la 1.000 de născuți vii.
     - *under_5_mortality* : numărul deceselor copiilor sub 5 ani la 1.000 de născuți vii.
     - *prev_hiv* : procentul populației cu vârsta cuprinsă între 15 și 49 de ani care trăiește cu HIV.
     """)

elif section == "Încărcare și afișare":
    st.header("Încărcare și afișare date")
    st.subheader("1. Încarcă documentul corespunzător:")
    uploaded_file = st.file_uploader("Încarcă un fișier", type=["csv"])
    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.session_state.file_name = uploaded_file.name
        st.write("Fișier încărcat:", uploaded_file.name)


    if st.session_state.df is not None:
        st.subheader("2. Afișarea datelor din fișier:")
        st.dataframe(st.session_state.df)
    else:
        st.warning("Nu a fost încărcat niciun fișier.")

elif section == "Selecție și filtrare":
    st.header("Selecția și filtrara datelor")

    if st.session_state.df is not None:
        st.write("Fișier utilizat:", st.session_state.file_name)
        df=st.session_state.df
        if "country" in df.columns:
            countries = df["country"].unique()
            selected_country = st.selectbox("Alege o țară:", countries, key="country_selectbox")

            if "year" in df.columns:
                st.subheader("1. Filtrare după țară și intervalul de ani:")
                min_year, max_year = int(df["year"].min()), int(df["year"].max())
                selected_year_range = st.slider("Alege intervalul de ani:", min_year, max_year, (min_year, max_year))

                filtered_df_year = df[(df["country"] == selected_country) & df["year"].between(*selected_year_range)]
                st.write("Rezultate filtrate:")
                st.dataframe(filtered_df_year)
            else:
                st.warning("Fișierul nu conține o coloană specifică anilor pentru filtrare.")

            st.subheader("2. Filtrare după țară și coloane:")
            selected_columns = st.multiselect("Alege coloanele:", df.columns.tolist(),
                                              default=["health_exp", "life_expect"])

            filtered_df_columns = df[df["country"] == selected_country][selected_columns]
            st.write("Rezultate filtrate:")
            st.dataframe(filtered_df_columns)

        else:
            st.warning("Fișierul nu conține o coloană ce permite filtrarea dupa 'country'.")
    else:
        st.warning("Încărcați mai întâi un fișier în secțiunea 'Încărcare și afișare'.")

elif section == "Vizualizare grafice":
    st.header("Vizualizare grafice")

    if st.session_state.df is not None:
        st.write("Fișier utilizat:", st.session_state.file_name)

        df = st.session_state.df
        if "country" in df.columns:
            st.subheader("1. Grafic comparație pentru o variabilă aleasă între țări:")
            countries = df["country"].unique()
            selected_countries = st.multiselect("Alege țările:", countries,
                                                default=countries[:2])

            all_variables_columns = [col for col in df.columns if col not in ["country", "year", "country_code"]]

            if all_variables_columns:
                selected_variable = st.selectbox("Alege variabila pentru comparație:", all_variables_columns)

                filtred_df_countries_variable = df[df["country"].isin(selected_countries)][
                    ["country", "year", selected_variable]]
                filtred_df_countries_variable = filtred_df_countries_variable.sort_values(by=["year"])
                st.line_chart(
                    filtred_df_countries_variable.pivot(index="year", columns="country", values=selected_variable))
            else:
                st.warning("Nu există variabile disponibile pentru analiză.")

            st.subheader("2. Histogramă pentru o variabilă într-o țară selectată:")
            countries = df["country"].unique()
            selected_country = st.selectbox("Alege o țară:", countries, key="country_selectbox")
            if all_variables_columns:
                selected_variable_histogram = st.selectbox("Alege variabila pentru histogramă:", all_variables_columns)

                filtered_df_histogram = df[df["country"] == selected_country][["year", selected_variable_histogram]]
                filtered_df_histogram = filtered_df_histogram.sort_values(by=["year"])
                st.bar_chart(filtered_df_histogram.set_index("year"))
            else:
                st.warning("Nu există variabile disponibile pentru analiză.")
    else:
        st.warning("Încărcați mai întâi un fișier în secțiunea 'Încărcare și afișare'.")

elif section =="🛠️ Prelucrarea datelor":
    st.header("🛠️ Prelucrarea datelor")
    st.markdown("""
    Datele din fișierul inițial vor fi prelucrate pentru a obține un model performant și pentru a evita erorile din fazele ulterioare.  
    Pentru a ajunge la scopul menționat, vom utiliza procesul *Exploratory Data Analysis (EDA)* prin intermediul căruia investigăm, explorăm și înțelegem datele înainte de a implementa modele predictive sau statistice.  

    **Obiectivele EDA includ:**  
    - Identificarea valorilor lipsă și tratarea acestora.  
    - Detectarea și gestionarea valorilor aberante (*outliers*).  
    - Vizualizarea și înțelegerea relațiilor dintre variabile.  
    - Identificarea variabilelor relevante pentru predicție.  
    """)

    st.subheader("Aflăm outlierii folosind Metoda IQR")
    st.write(
        """Ce este Metoda IQR?  
        Cea mai uzuală metodă pentru date cu distribuție necunoscută (adesea asimetrică) este IQR (Interquartile Range).  
        IQR = Q3 − Q1  
        Q1 = Quartila 1 (25% din date sunt sub Q1).  
        Q3 = Quartila 3 (75% din date sunt sub Q3).  
        Limite:  
        lower_bound = Q1 − 1.5×IQR  
        upper_bound = Q3 + 1.5×IQR  
        Valorile care se află în afara acestui interval [lower_bound, upper_bound] sunt considerate outlieri potențiali.""")


elif section == "Analiza valorilor":
    st.header("Analiza detaliată a valorilor")
    if st.session_state.df is not None:
        # stergem coloanele irelevante
        df_initial = st.session_state.df.copy()
        df_mod = df_initial.drop(columns=['country','country_code', 'year'])

        st.subheader("1. Analiza statistică descriptivă:")
        st.dataframe(df_mod.describe())

        st.subheader("2. Analiza valorilor lipsă:")
        st.write("Înainte de a trece la tratarea valorilor lipsă este important să identificăm unde există valori lipsă")

        missing_values_df=analiza_valorilor_lipsa(df_mod)
        st.dataframe(missing_values_df)

        st.subheader("3. Distribuția valorilor lipsă:")
        fig_valori_lipsa=plot_bar_valori_lipsa(missing_values_df)
        st.pyplot(fig_valori_lipsa)

    else:
        st.warning("Încărcați mai întâi un fișier în secțiunea 'Încărcare și afișare'.")

elif section == "Tratarea valorilor lipsă":
    st.header("Tratarea valorilor lipsă")
    if st.session_state.df is not None:
        df_tratat = st.session_state.df.copy()

        st.subheader("1) Imputare cu media globală: life_expect")
        st.write("""
                În urma analizei setului de date, s-a observat că valorile lipsă pentru *life_expect* apar *doar pentru țara Monaco*.
                Pentru a evita excluderea acesteia din analiză, valorile lipsă au fost înlocuite cu *media globală* din restul țărilor.
                """)

        valori_nan_inainte = df_tratat[df_tratat["country"] == "Monaco"]['life_expect'].isna().sum()
        media_globala = df_tratat['life_expect'].mean()
        df_tratat['life_expect'] = df_tratat['life_expect'].fillna(media_globala)
        valori_nan_dupa = df_tratat[df_tratat["country"] == "Monaco"]['life_expect'].isna().sum()

        st.write(f"Monaco avea **{valori_nan_inainte} valori lipsă** pentru *life_expect* înainte de tratare.După înlocuirea cu media globală ({media_globala:.2f}), au rămas **{valori_nan_dupa} valori lipsă**.")

        st.subheader("2) Propagare și eliminare țări: health_exp")
        st.write("""
               Pentru variabila health_exp s-a aplicat propagare înainte și înapoi pe fiecare țară.
                Apoi, s-au eliminat complet țările fără nicio valoare disponibilă în întregul interval analizat. """)

        st.markdown("**Valori înainte de tratare**")
        st.dataframe(
            df_tratat[df_tratat['health_exp'].isna()][["country", "year", "health_exp"]]
        )
        df_tratat['health_exp'] = df_tratat.groupby('country')['health_exp'].transform(lambda x: x.bfill().ffill())
        tari_de_eliminat = ["Korea, Dem. People's Rep.", "West Bank and Gaza", "Somalia"]
        df_tratat = df_tratat[~df_tratat['country'].isin(tari_de_eliminat)]
        st.markdown("**Valori după tratare**")
        st.dataframe(
            df_tratat[["country", "year", "health_exp"]]
        )

        st.subheader("3) Imputare KNN: maternal_mortality")
        st.write("""
        Pentru maternal_mortality s-a constatat că valorile lipsă apar în principiu pentru același număr de aproximativ 5 țări pe toata perioada analizată.
        Cum nu exista suficiente date pentru tratare prin metode precum interpolarea sau propagarea valorilor s-a decis tratarea prin KNN Imputation (k-Nearest Neighbors)
        ceastă metodă estimează valorile lipsă pe baza valorilor altor țări cu profil similar folosind variabile precum: health_exp si life_expect
        """)

        df_aux = df_tratat[['maternal_mortality', 'health_exp', 'life_expect']].copy()
        imputer = KNNImputer(n_neighbors=5)
        df_imputed = imputer.fit_transform(df_aux)

        df_tratat['maternal_mortality'] = df_imputed[:, 0]
        tari_lipsa_initial = ['Monaco', 'Dominica', 'St. Kitts and Nevis', 'Nauru', 'Marshall Islands','Andorra']

        st.markdown("**Valori lipsă înainte de tratare**")
        st.dataframe(
            st.session_state.df[
                (st.session_state.df['country'].isin(tari_lipsa_initial)) &
                (st.session_state.df['maternal_mortality'].isna())
                ][['country', 'year', 'maternal_mortality']]
        )

        st.markdown("**Valori după imputare (KNN)**")
        st.dataframe(
            df_tratat[df_tratat['country'].isin(tari_lipsa_initial)][['country', 'year', 'maternal_mortality']]
        )
        st.subheader("4) Imputare KNN: prev_undernourishment")
        st.write("""
        Variabila prev_undernourishment are un procentaj semnificativ de valori lipsă (11%).
        Cum valorile lipsă sunt corespunzătoare acelorași țări pentru perioada analizată s-a ales ca metoda de tratare KNN Imputation .
        Aceasta completează valorile lipsă pe baza celor mai apropiate observații (țări) din punct de vedere al altor variabile relevante, cum ar fi:health_exp,life_expect,under_5_mortality.
        Astfel se va menține coerența statistică și observațiile vor fi păstrate în setul de date.
        """)

        lipsa_inainte = df_tratat['prev_undernourishment'].isna().sum()
        cols = ['prev_undernourishment', 'health_exp', 'life_expect', 'under_5_mortality']
        df_tratat[cols] = KNNImputer(n_neighbors=5).fit_transform(df_tratat[cols])
        lipsa_dupa = df_tratat['prev_undernourishment'].isna().sum()

        st.write(f"prev_undernourishment avea {lipsa_inainte} valori lipsă înainte de tratare.")
        st.write(f"După interpolare liniară, au rămas {lipsa_dupa} valori lipsă.")

        st.subheader("4) Eliminare coloana: prev_hiv")
        df_tratat.drop(columns=['prev_hiv'], inplace=True)
        st.write(
            "Coloana `prev_hiv` a fost eliminată, întrucât are un procent ridicat de valori lipsă și nu este esențială pentru analiză.")

        ##### de aici in jos nu modificam
        st.session_state.df_prelucrat = df_tratat #salvare date prelucrate

        st.text("✅ După tratarea valorilor lipsă nu mai există valori lipsă in setul de date:")
        cols_to_drop = ['country', 'country_code', 'year']
        st.dataframe(analiza_valorilor_lipsa(df_tratat.drop(columns=cols_to_drop)))
        ###pana aici

    else:
        st.warning("Încărcați mai întâi un fișier în secțiunea 'Încărcare și afișare'.")

elif section == "Prelucrarea avansată a datelor":
    st.header("Prelucrarea avansată a datelor")
    if st.session_state.df_prelucrat is not None:
        df = st.session_state.df_prelucrat
        st.subheader("1) Vizualizarea prin histograme a datelor tratate")
        histograme_variabile_pt_outlieri=histograme_variabile(df)
        st.pyplot(histograme_variabile_pt_outlieri)

        st.write("""
        **Pentru health_exp**:  Distribuția este puternic asimetrică spre dreapta (skewed right). Majoritatea valorilor sunt concentrate în intervalul mai mic (2.5-5), cu o coadă lungă spre valori mai mari. Acest lucru sugerează că multe țări au cheltuieli relativ scăzute pentru sănătate, în timp ce un număr mai mic de țări au cheltuieli semnificativ mai mari.
        
        **Pentru life_expect**: Distribuția pare ușor asimetrică spre stânga (skewed left) sau poate fi considerată o distribuție bimodală, cu două vârfuri. Acest lucru ar putea indica grupuri de țări cu speranță de viață diferită, posibil din cauza diferențelor în dezvoltarea economică, accesul la servicii medicale etc.
        
        **Pentru maternal_mortality**: Distribuția este extrem de asimetrică spre dreapta (skewed right). Majoritatea valorilor sunt concentrate în intervalul mai mic, cu o coadă lungă spre valori foarte mari. Aceasta indică faptul că mortalitatea maternă este relativ scăzută în majoritatea țărilor, dar există un număr semnificativ de țări cu rate foarte mari.
        
        **Pentru infant_mortality**: Distribuția este asimetrică spre dreapta (skewed right), similar cu mortalitatea maternă. Majoritatea valorilor sunt scăzute, cu o coadă lungă spre valori mai mari. Acest lucru sugerează că, deși mortalitatea infantilă este general scăzută, există încă țări cu rate îngrijorător de mari.
        
        **Pentru neonatal_mortality**: Distribuția este asimetrică spre dreapta (skewed right), similar cu mortalitatea infantilă. Concentrarea valorilor în intervalul mai mic indică o mortalitate neonatală relativ scăzută în majoritatea țărilor, dar cu variații semnificative.
        
        **Pentru under_5_mortality**: Distribuția este asimetrică spre dreapta (skewed right). Similar cu mortalitatea infantilă și neonatală, majoritatea valorilor sunt scăzute, cu o coadă lungă spre valori mai mari.
        
        **Pentru inci_tuberc**: Distribuția este extrem de asimetrică spre dreapta (skewed right). Majoritatea valorilor sunt concentrate în intervalul mai mic, cu o coadă lungă spre valori foarte mari. Acest lucru indică faptul că incidența tuberculozei este relativ scăzută în majoritatea țărilor, dar există un număr semnificativ de țări cu rate foarte mari.
        
        **Pentru prev_undernourishment**: Distribuția pare să fie asimetrică spre dreapta (skewed right). Majoritatea țărilor au o prevalență scăzută a subnutriției, dar există o coadă lungă spre valori mai mari, indicând țări cu probleme semnificative de subnutriție.
                """)

        st.subheader("2) Găsirea outlierilor folosind Metoda IQR")
        outliers = gaseste_outlieri_iqr(df)

        # Iterăm prin fiecare coloană și afișăm rezultatele
        for col, (lower_bound, upper_bound, outliers_df) in outliers.items():
            # Creăm figura pentru boxplot
            fig_outlieri, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x=df[col], ax=ax)
            ax.set_title(f"Boxplot pentru '{col}'")
            ax.set_xlabel(col)
            plt.tight_layout()

            st.pyplot(fig_outlieri)

            #Afișăm detalii despre outlieri
            st.write(f"Coloana: {col}")
            st.write(f"Limita inferioară: {lower_bound:.2f}, Limita superioară: {upper_bound:.2f}")
            st.write(f"Număr de outlieri: {len(outliers_df)}")
            st.write("-" * 50)

        st.subheader("3) Tratarea outlierilor")
        st.write("""
            **Tratarea outlierilor a fost efectuată astfel:**

            - **health_exp:** Valorile mai mici decât limita inferioară sau mai mari decât limita superioară (calculată cu metoda IQR) au fost înlocuite cu valoarea limitei respective (Winsorization).

            - **life_expect:** Valorile sub limita inferioară au fost înlocuite cu limita inferioară calculată.

            - **maternal_mortality:** S-a aplicat o transformare logaritmică (log(x + 1)) pentru a comprima diferențele dintre valorile foarte mari și restul datelor.

            - **infant_mortality:** Valorile din partea superioară (peste percentila 99) au fost trunchiate la valoarea de la percentila 99.

            - **neonatal_mortality:** Nu s-a aplicat niciun tratament deoarece nu au fost identificați outlieri.

            - **under_5_mortality:** Valorile peste percentila 99 au fost trunchiate.

            - **inci_tuberc:** Valorile peste percentila 99 au fost winsorizate (înlocuite cu valoarea de la percentila 99).

            - **prev_undernourishment:** Valorile peste percentila 99 au fost winsorizate.
        """)
        df_tratare_outlieri = tratare_outlieri(df)

        # Iterăm prin fiecare coloană din df_tratare_outlieri și afișăm boxplot-urile și detaliile
        outliers_nou = gaseste_outlieri_iqr(df_tratare_outlieri)
        for col, (lower_bound, upper_bound, outliers_df) in outliers_nou.items():
            # Creăm figura pentru boxplot
            fig_outlieri_nou, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x=df_tratare_outlieri[col], ax=ax, color='lightgreen')
            ax.set_title(f"Boxplot după tratamentul outlierilor pentru '{col}'")
            ax.set_xlabel(col)
            plt.tight_layout()

            st.pyplot(fig_outlieri_nou)

            # Afișăm detalii despre outlieri în noul DataFrame
            st.write(f"Coloana: {col}")
            st.write(f"Limita inferioară: {lower_bound:.2f}, Limita superioară: {upper_bound:.2f}")
            st.write(f"Număr de outlieri în noul DataFrame: {len(outliers_df)}")
            st.write("-" * 50)

        st.write("Histograme după tratarea outlierilor:")
        histograme_tratate = histograme_variabile(df_tratare_outlieri)
        st.pyplot(histograme_tratate)

        st.write("""
        După aplicarea tratamentului, nu mai se identifică outlieri semnificativi în setul de date, 
        ceea ce indică că intervențiile aplicate (winsorizare, trunchiere și transformări logaritmice) 
        au reușit să reducă impactul valorilor extreme asupra distribuțiilor. Mai departe, după ce am tratat outlierii 
        și am aplicat transformări, următorii pași în analiza exploratorie a datelor (EDA) includ analiza corelațiilor 
        între variabile, extragerea de noi caracteristici (feature engineering) și pregătirea datelor pentru modelare.
        """)

        st.subheader("4) Analiza corelațiilor între variabile")
        # Pentru noi toate datele sunt numerice momentan deci afisam toate coloanele numerice, in afara de YEAR
        matrice_corelatie = afiseaza_matrice_corelatie(df_tratare_outlieri)
        st.pyplot(matrice_corelatie)

        st.subheader("5) Standardizare și normalizare")
        st.write("Acum trecem la standardizare și normalizare, două tehnici esențiale de scalare a datelor pentru a pregăti variabilele numerice "
                 "înainte de a le folosi în modele de machine learning. Aceste metode ajută la reducerea diferențelor de scară între variabile "
                 "și la îmbunătățirea performanței anumitor algoritmi")
        # st.write("**Standardizarea** transformă datele astfel încât media să fie 0 și deviația standard 1 (Z-score). ")
        st.write("**Normalizarea** (Min-Max Scaling) aduce toate valorile într-un interval standard, de obicei [0, 1].")

        numerical_cols = [col for col in df.select_dtypes(include=[np.number]).columns if
                          col.upper() != 'YEAR'.upper()]
        scaler_norm = MinMaxScaler()
        df_final = df_tratare_outlieri.copy()
        df_final[numerical_cols] = scaler_norm.fit_transform(df_tratare_outlieri[numerical_cols])

        st.write("**SETUL DE DATE FINAL**")
        st.dataframe(df_final.head())

        st.subheader("6) Prezicerea variabilei țintă")

        target = 'life_expect'

        X = df_final.drop(columns=[target, 'country', 'country_code', 'year'])
        y = df_final[target]

        X = X.select_dtypes(include=[np.number])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f"Rezultatele modelului de regresie liniară pentru {target}:")
        st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
        st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
        st.write(f"**R2 Score:** {r2:.4f}")

        st.subheader("7) Comparație modele: Random Forest vs XGBoost")
        target = 'life_expect'

        X = df_final.drop(columns=[target, 'country', 'country_code', 'year'])  # eliminăm coloanele nerelevante
        y = df_final[target]

        X = X.select_dtypes(include=[np.number])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)

        xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)

        rf_mae = mean_absolute_error(y_test, rf_pred)
        rf_mse = mean_squared_error(y_test, rf_pred)
        rf_r2 = r2_score(y_test, rf_pred)

        xgb_mae = mean_absolute_error(y_test, xgb_pred)
        xgb_mse = mean_squared_error(y_test, xgb_pred)
        xgb_r2 = r2_score(y_test, xgb_pred)


        col1, col2 = st.columns(2)

        with col1:
            st.markdown("Random Forest")
            st.write("*MAE*:", f"{rf_mae:.4f}")
            st.write("*MSE*:", f"{rf_mse:.4f}")
            st.write("*R²*:", f"{rf_r2:.4f}")

        with col2:
            st.markdown("XGBoost")
            st.write("*MAE*:", f"{xgb_mae:.4f}")
            st.write("*MSE*:", f"{xgb_mse:.4f}")
            st.write("*R²*:", f"{xgb_r2:.4f}")

        #SALVEAZA DATELE DUPA TRATARE IN DF_PRELUCRAT!!! CA MAI SUS
        st.session_state.df_prelucrat = df_final
    else:
        st.warning("Încărcați mai întâi un fișier în secțiunea 'Încărcare și afișare'. Ulterior treceți prin secțiunea ”Tratarea valorilor lipsă” pentru a putea continua.")





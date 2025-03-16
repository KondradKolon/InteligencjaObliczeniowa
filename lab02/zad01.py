import pandas as pd
import numpy as np
df = pd.read_csv('iris_with_errors.csv')
#A)

#wyswietlanie danych z bledami
print(df)

total_mistakes_count = df.isna().sum()
print(total_mistakes_count)

#Policzenie ile jest w bazie brakujacy lub nieuzupelnionych danych i  zamiana na NaN
df['sepal.length'] = pd.to_numeric(df['sepal.length'], errors='coerce')
df['sepal.width'] = pd.to_numeric(df['sepal.width'], errors='coerce')
df['petal.length'] = pd.to_numeric(df['petal.length'], errors='coerce')
df['petal.width'] = pd.to_numeric(df['petal.width'], errors='coerce')

print(df)
total_mistakes_count = df.isna().sum()
print(total_mistakes_count)
kolumny_liczbowe = df.select_dtypes(include=['number']).columns
print("kolumny_liczbowe:", kolumny_liczbowe)

for col in kolumny_liczbowe:
    mean=df[col].mean()
    df[col] = [mean if x <= 0 or x > 15 else x for x in df[col]]
#zamiana z poza zasiegu na srednia 


kolumna_rodzaj = df.columns[-1]  
kolumna_rodzaj_unique = df[kolumna_rodzaj].unique()
print("wszystkie stringi:", kolumna_rodzaj_unique)

# sprawdzamy jakie sa zle wartosci
bledne_wartosci = set(kolumna_rodzaj_unique) - {"Setosa", "Versicolor", "Virginica"}
print("Incorrect variety names:", bledne_wartosci)    

def correct_variety_names(name):
    
    name = name.lower()  
    if 'setosa' in name:
        return 'Setosa'
    elif 'versicolor' in name or 'versicolour' in name:
        return 'Versicolor'
    elif 'virginica' in name:
        return 'Virginica'
    else:
        return name  

df[kolumna_rodzaj] = df[kolumna_rodzaj].map(correct_variety_names)

print("rodzaje po korekcie :", df[kolumna_rodzaj].unique())
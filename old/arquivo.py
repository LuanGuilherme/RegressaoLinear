import pandas as pd
from IPython.display import HTML
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import numpy as np
from collections import Counter
import scipy.stats as stats
import seaborn as sns
import statsmodels.formula.api as sm
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression as skl
from sklearn.metrics import mean_squared_error
import statsmodels.api as sme

# tratamento do dataset
dadosAustralia = pd.read_csv(
    "Crash_Data.csv", sep=",", low_memory=False)  # lendo .csv do dataset
dadosAustralia = dadosAustralia.loc[:, [
    'Speed Limit', 'Time', 'Gender', 'National Remoteness Areas','Age','Year', 'Month', 'Crash Type']]  # selecionando variáveis
dadosAustralia.rename(columns={'Speed Limit': 'VelocidadeMaxima', 'Time': 'Horário', 'Year': 'Ano',
                      'Gender': 'Gênero', 'National Remoteness Areas': 'Região', 'Age': 'Idade', 'Month': 'Mes','Crash Type': 'TipoBatida'}, inplace=True)  # renomeando variáveis

dadosAustralia = dadosAustralia.dropna()  # tratando valores faltantes
dadosAustralia.apply(lambda x: pd.to_numeric(
    x, errors='coerce')).dropna()  # tratando valores faltantes
dadosAustralia = dadosAustralia[dadosAustralia["VelocidadeMaxima"].str.contains(
    "<") == False]  # tratando valores incompletos
dadosAustralia = dadosAustralia[dadosAustralia["VelocidadeMaxima"].str.contains(
    "Unspecified") == False]  # tratando valores incompletos

dadosAustralia['Idade'] = dadosAustralia['Idade'].astype(int)


# convertendo horário string para minutos int
for index, row in dadosAustralia.iterrows():
    numeros = row['Horário'].split(':')
    dadosAustralia['Horário'][index] = (
        int(numeros[0]) * 60) + (int(numeros[1]))

# cálculo de médias, modas e medianas

medianaHorarioMinutos = dadosAustralia['Horário'].median()
medianaHorario = str(int(medianaHorarioMinutos // 60)).rjust(2, '0') + \
    ':' + str(int(medianaHorarioMinutos % 60)).rjust(2, '0')

modaHorarioMinutos = dadosAustralia['Horário'].mode()
modasHorarios = []

for modaHorarioMinutos in modaHorarioMinutos:
    modasHorarios.append(str(int(modaHorarioMinutos // 60)).rjust(2, '0') +
                         ':' + str(int(modaHorarioMinutos % 60)).rjust(2, '0'))

mediaHorarioMinutos = dadosAustralia['Horário'].mean()
mediaHorario = str(int(mediaHorarioMinutos // 60)).rjust(2, '0') + \
    ':' + str(int(mediaHorarioMinutos % 60)).rjust(2, '0')

pQuartilHora = dadosAustralia['Horário'].quantile(0.25)
pQuartilHora = str(int(pQuartilHora // 60)).rjust(2, '0') + \
    ':' + str(int(pQuartilHora % 60)).rjust(2, '0')

sQuartilHora = dadosAustralia['Horário'].quantile(0.75)
sQuartilHora = str(int(sQuartilHora // 60)).rjust(2, '0') + \
    ':' + str(int(sQuartilHora % 60)).rjust(2, '0')

dadosAustralia['VelocidadeMaxima'] = pd.to_numeric(
    dadosAustralia['VelocidadeMaxima'])  # convertendo vel maxima para numerico

medianaVelocidade = dadosAustralia['VelocidadeMaxima'].median()
modasVelocidade = dadosAustralia['VelocidadeMaxima'].mode()
mediaVelocidade = dadosAustralia['VelocidadeMaxima'].mean()

pQuartilVelocidade = dadosAustralia['VelocidadeMaxima'].quantile(0.25)
sQuartilVelocidade = dadosAustralia['VelocidadeMaxima'].quantile(0.75)

modasGenero = dadosAustralia['Gênero'].mode()

modasRegiao = dadosAustralia['Região'].mode()

valoresMediana = {'Horário':  [medianaHorario],
                  'VelocidadeMaxima(km/h)': [medianaVelocidade]
                  }
valoresModa = {'Horário':  modasHorarios,
               'VelocidadeMaxima(km/h)': modasVelocidade,
               'Gênero': modasGenero,
               'Região': modasRegiao
               }
valoresMedia = {'Horário':  [mediaHorario],
                'VelocidadeMaxima(km/h)': [mediaVelocidade]
                }
valoresQuartisHora = {  '1º quartil': [pQuartilHora],
                        '3º quartil': [sQuartilHora]
                    }
valoresQuartisVelocidade = {    '1º quartil': [pQuartilVelocidade],
                                '3º quartil': [sQuartilVelocidade]
                            }

# RENDERIZAÇÃO DE TABELAS PARA RELATÓRIO


print('\033[1mMedianas\033[0m')
print(pd.DataFrame(valoresMediana).to_string(index=False))
HTML(pd.DataFrame(valoresMediana).to_html(index=False))
HTML(pd.DataFrame(valoresMediana).style.set_table_styles(
    [{"selector": "", "props": [("border", "1px solid grey")]},
      {"selector": "tbody td", "props": [("border", "1px solid grey")]},
     {"selector": "th", "props": [("border", "1px solid grey")]}
    ]
).hide_index().to_html())

print()

print('\033[1mModas\033[0m')
print(pd.DataFrame(valoresModa).to_string(index=False))
HTML(pd.DataFrame(valoresModa).to_html(index=False))
HTML(pd.DataFrame(valoresModa).style.set_table_styles(
    [{"selector": "", "props": [("border", "1px solid grey")]},
      {"selector": "tbody td", "props": [("border", "1px solid grey")]},
     {"selector": "th", "props": [("border", "1px solid grey")]}
    ]
).hide_index().to_html())

print()

print('\033[1mMédias\033[0m')
print(pd.DataFrame(valoresMedia).to_string(index=False))
HTML(pd.DataFrame(valoresMedia).to_html(index=False))
HTML(pd.DataFrame(valoresMedia).style.set_table_styles(
    [{"selector": "", "props": [("border", "1px solid grey")]},
      {"selector": "tbody td", "props": [("border", "1px solid grey")]},
     {"selector": "th", "props": [("border", "1px solid grey")]}
    ]
).hide_index().to_html())

print()

print('\033[1mQuartis de horário\033[0m')
print(pd.DataFrame(valoresQuartisHora).to_string(index=False))
HTML(pd.DataFrame(valoresQuartisHora).to_html(index=False))
HTML(pd.DataFrame(valoresQuartisHora).style.set_table_styles(
    [{"selector": "", "props": [("border", "1px solid grey")]},
      {"selector": "tbody td", "props": [("border", "1px solid grey")]},
     {"selector": "th", "props": [("border", "1px solid grey")]}
    ]
).hide_index().to_html())

print()

print('\033[1mQuartis de VelocidadeMaxima\033[0m')
print(pd.DataFrame(valoresQuartisVelocidade).to_string(index=False))
HTML(pd.DataFrame(valoresQuartisVelocidade).to_html(index=False))
HTML(pd.DataFrame(valoresQuartisVelocidade).style.set_table_styles(
    [{"selector": "", "props": [("border", "1px solid grey")]},
      {"selector": "tbody td", "props": [("border", "1px solid grey")]},
     {"selector": "th", "props": [("border", "1px solid grey")]}
    ]
).hide_index().to_html())

# gerar tabelas no google colab
# df = pd.DataFrame(valoresModa)
# mean_df = df.mean()
# display(df.style.hide_index().set_caption("Modas"))

desvioPadraoVelocidade = dadosAustralia['VelocidadeMaxima'].std()
varianciaVelocidade = dadosAustralia['VelocidadeMaxima'].var()
coeficienteVariacaoVelocidade = (desvioPadraoVelocidade / mediaVelocidade) * 100

desvioPadraoHorarioMinutos = dadosAustralia['Horário'].std()
varianciaHorario = dadosAustralia['Horário'].var()
coeficienteVariacaoHorario = (desvioPadraoHorarioMinutos/mediaHorarioMinutos) * 100

desvioPadraoHorario = str(int(desvioPadraoHorarioMinutos // 60)).rjust(2, '0') + \
    ':' + str(int(desvioPadraoHorarioMinutos % 60)).rjust(2, '0')

valoresDesvioPadrao = {'Horário':  [desvioPadraoHorario],
                       'VelocidadeMaxima(km/h)': [desvioPadraoVelocidade]
                       }
valoresVarianca = {'Horário':  [varianciaHorario],
                   'VelocidadeMaxima': [varianciaVelocidade]
                   }
valoresCoeficienteVariacao = {'Horário(%)':  [coeficienteVariacaoHorario],
                              'VelocidadeMaxima(%)': [coeficienteVariacaoVelocidade]
                              }

# RENDERIZAÇÃO DE TABELAS PARA RELATÓRIO

print()

print('\033[1mDesvio Padrão\033[0m')
print(pd.DataFrame(valoresDesvioPadrao).to_string(index=False))
HTML(pd.DataFrame(valoresDesvioPadrao).to_html(index=False))
HTML(pd.DataFrame(valoresDesvioPadrao).style.set_table_styles(
    [{"selector": "", "props": [("border", "1px solid grey")]},
     {"selector": "tbody td", "props": [("border", "1px solid grey")]},
     {"selector": "th", "props": [("border", "1px solid grey")]}
     ]
).hide_index().to_html())

print()

print('\033[1mVariância\033[0m')
print(pd.DataFrame(valoresVarianca).to_string(index=False))
HTML(pd.DataFrame(valoresVarianca).to_html(index=False))
HTML(pd.DataFrame(valoresVarianca).style.set_table_styles(
    [{"selector": "", "props": [("border", "1px solid grey")]},
     {"selector": "tbody td", "props": [("border", "1px solid grey")]},
     {"selector": "th", "props": [("border", "1px solid grey")]}
     ]
).hide_index().to_html())

print()

print('\033[1mCoeficiente de Variação\033[0m')
print(pd.DataFrame(valoresCoeficienteVariacao).to_string(index=False))
HTML(pd.DataFrame(valoresCoeficienteVariacao).to_html(index=False))
HTML(pd.DataFrame(valoresCoeficienteVariacao).style.set_table_styles(
    [{"selector": "", "props": [("border", "1px solid grey")]},
     {"selector": "tbody td", "props": [("border", "1px solid grey")]},
     {"selector": "th", "props": [("border", "1px solid grey")]}
     ]
).hide_index().to_html())


# GERANDO HISTOGRAMAS GENERO
plt.figure(figsize=(10,7))
pd.Series(dadosAustralia['Gênero']).value_counts(sort=False).plot(kind='bar')
plt.title('Distribuição de acidentes fatais por gênero')
plt.xlabel('Gênero (masculino/feminino)')
plt.xticks(rotation=1)
plt.ylabel('Acidentes fatais')
plt.savefig('histograma-genero.png')
plt.show()
plt.close()

# GERANDO HISTOGRAMAS VELOCIDADE MÃXIMA

plt.figure(figsize=(8,5))
plt.hist(dadosAustralia['VelocidadeMaxima'], rwidth=0.9, bins=12)
plt.title('Distribuição de acidentes fatais por VelocidadeMaxima da rodovia')
plt.xlabel('Velocidade (em km/h)')
x = np.arange(0, 131, 10)
plt.xticks(x)
plt.ylabel('Acidentes fatais')
plt.savefig('histograma-velocidade-maxima.png')
plt.show()
plt.close()

# GERANDO HISTOGRAMAS HORÁRIO

'''plt.figure(figsize=(10,7), dpi= 80)
sns.distplot(dadosAustralia["VelocidadeMaxima"], color="dodgerblue", label="Compact",)
plt.savefig('normal.png')
plt.show()
plt.close()'''

# convertendo horário de minutos para hora
horarioEmMinutos = []
for index, row in dadosAustralia.iterrows():
    numero = row['Horário']
    horarioEmMinutos.append(numero/60)

dadosAustralia.insert(0, 'Horário (em h)', horarioEmMinutos)
plt.figure(figsize=(8,5))
plt.hist(horarioEmMinutos, rwidth=0.9, bins=range(1,24))
plt.title('Distribuição de acidentes fatais por horário')
plt.xlabel('Horário (em h)')
x = np.arange(0, 24, 1)
plt.xticks(x)
plt.ylabel('Acidentes fatais')
plt.savefig('histograma-horarios.png')
plt.show()
plt.close()

# GERANDO HISTOGRAMAS REGIÃO
plt.figure(figsize=(10,7))
pd.Series(dadosAustralia['Região']).value_counts(sort=False).plot(kind='bar')
plt.title('Distribuição de acidentes fatais por região'),
plt.xlabel('Região')
plt.xticks(rotation=15)
plt.ylabel('Acidentes fatais')
plt.savefig('histograma-regiao.png')
plt.show()
plt.close()

# BOXPLOT VELOCIDADE 
boxplot = dadosAustralia.boxplot(column="VelocidadeMaxima", by="Gênero")
boxplot.plot()
plt.title('Distribuição de acidentes fatais: VelocidadeMaxima por gênero')
plt.ylabel('Velocidade (km/h)')
plt.suptitle('')
plt.savefig('velocidade-por-genero.png')
plt.show()
plt.close()

# BOXPLOT VELOCIDADE 
boxplot = dadosAustralia.boxplot(column="Horário (em h)", by="Região")
boxplot.plot()
plt.title('Distribuição de acidentes fatais: horário por região')
plt.ylabel('Horário')
plt.xticks(rotation=15)
plt.suptitle('')
plt.savefig('horario-por-regiao.png')
plt.show()
plt.close()

dadosAustralia.rename(columns={'Horário (em h)': 'HorarioH'}, inplace=True)  # renomeando variáveis

# print(stats.normaltest(dadosAustralia['VelocidadeMaxima']))
dadosAustralia['HorarioH'] = dadosAustralia['HorarioH'].astype(int)

amostra = dadosAustralia.sample(1000)
plt.figure(figsize = (16,8))
plt.scatter(
    amostra['Horário'], 
    amostra['Idade'], 
    c='red')
plt.xlabel("Idade")
plt.ylabel("HorarioH")
plt.show()
plt.savefig('dispersao')




X = amostra['Mes'].values.reshape(-1,1)
y = amostra['VelocidadeMaxima'].values.reshape(-1,1)
reg = skl()
reg.fit(X, y)
f_previsaoes = reg.predict(X)
plt.figure(figsize = (16,8))
plt.scatter(
    amostra['Mes'], 
    amostra['VelocidadeMaxima'], 
    c='red')
plt.xlabel("Mes")
plt.ylabel("Velocidade máxima")
plt.show()
plt.savefig('dispersao-mes-velocidade')


###################
plt.figure(figsize = (16,8))
plt.scatter(
    amostra['Horário'], 
    amostra['Idade'], 
    c='red')
plt.xlabel("Idade")
plt.ylabel("HorarioH")
plt.show()
plt.savefig('dispersao')
X = amostra['Idade'].values.reshape(-1,1)
y = amostra['HorarioH'].values.reshape(-1,1)
reg = skl()
reg.fit(X, y)

f_previsaoes = reg.predict(X)


plt.figure(figsize = (16,8))
plt.scatter(
    amostra['Idade'], 
    amostra['HorarioH'], 
    c='red')


plt.plot(
    amostra['Idade'],
    f_previsaoes,
    c='blue',
    linewidth=3,
    linestyle=':'
)


plt.xlabel("Idade")
plt.ylabel("Horário (em h)")
plt.show()
plt.savefig('dispersao-idade-horario')

print(dadosAustralia)

modelo1 = sm.ols(formula='VelocidadeMaxima~Idade+Mes', data=amostra).fit()
print(modelo1.summary())
y = 86.1635 + (-0.0962 *amostra['Idade']) + (0.1838 * amostra['Mes'])
residuos = modelo1.predict()
sns.histplot(y-residuos, kde=True);
plt.title("Histograma dos resíduos Vel ~ Idade + Mes")
plt.savefig("residuos_velocidade_idade_mes")

modelo2 = sm.ols(formula='VelocidadeMaxima~Mes+HorarioH', data=amostra).fit()
print(modelo2.summary())
residuos = modelo2.predict()

y = 83.0886 + (0.1973 *amostra['Mes']) + (-0.1026 * amostra['HorarioH'])
sns.histplot(y-residuos, kde=True)
plt.title("Histograma dos resíduos Vel ~ Mes + HorarioH")
plt.savefig("residuos_velocidade_mes_horario")

modelo3 = sm.ols(formula='HorarioH~VelocidadeMaxima+Ano', data=amostra).fit()
print(modelo3.summary())
residuos = modelo3.predict()
y = 51.6436 + (-0.0076 *amostra['VelocidadeMaxima']) + (-0.0190 * amostra['Ano'])
sns.histplot(y-residuos, kde=True)
plt.title("Histograma dos resíduos Horario ~ Velocidade + ano")
plt.savefig("residuos_horario_velocidade_ano")


dadosAustralia.to_csv("teste.csv", index=False)
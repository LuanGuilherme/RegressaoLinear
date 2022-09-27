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
# tratamento do dataset
dadosAustralia = pd.read_csv(
    "Crash_Data.csv", sep=",", low_memory=False)  # lendo .csv do dataset
dadosAustralia = dadosAustralia.loc[:, [
    'Speed Limit', 'Time', 'Gender', 'National Remoteness Areas','Age']]  # selecionando variáveis
dadosAustralia.rename(columns={'Speed Limit': 'Velocidade máxima', 'Time': 'Horário',
                      'Gender': 'Gênero', 'National Remoteness Areas': 'Região', 'Age': 'Idade'}, inplace=True)  # renomeando variáveis

dadosAustralia = dadosAustralia.dropna()  # tratando valores faltantes
dadosAustralia.apply(lambda x: pd.to_numeric(
    x, errors='coerce')).dropna()  # tratando valores faltantes
dadosAustralia = dadosAustralia[dadosAustralia["Velocidade máxima"].str.contains(
    "<") == False]  # tratando valores incompletos
dadosAustralia = dadosAustralia[dadosAustralia["Velocidade máxima"].str.contains(
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

dadosAustralia['Velocidade máxima'] = pd.to_numeric(
    dadosAustralia['Velocidade máxima'])  # convertendo vel maxima para numerico

medianaVelocidade = dadosAustralia['Velocidade máxima'].median()
modasVelocidade = dadosAustralia['Velocidade máxima'].mode()
mediaVelocidade = dadosAustralia['Velocidade máxima'].mean()

pQuartilVelocidade = dadosAustralia['Velocidade máxima'].quantile(0.25)
sQuartilVelocidade = dadosAustralia['Velocidade máxima'].quantile(0.75)

modasGenero = dadosAustralia['Gênero'].mode()

modasRegiao = dadosAustralia['Região'].mode()

valoresMediana = {'Horário':  [medianaHorario],
                  'Velocidade Máxima(km/h)': [medianaVelocidade]
                  }
valoresModa = {'Horário':  modasHorarios,
               'Velocidade Máxima(km/h)': modasVelocidade,
               'Gênero': modasGenero,
               'Região': modasRegiao
               }
valoresMedia = {'Horário':  [mediaHorario],
                'Velocidade Máxima(km/h)': [mediaVelocidade]
                }
valoresQuartisHora = {  '1º quartil': [pQuartilHora],
                        '3º quartil': [sQuartilHora]
                    }
valoresQuartisVelocidade = {    '1º quartil': [pQuartilVelocidade],
                                '3º quartil': [sQuartilVelocidade]
                            }

# RENDERIZAÇÃO DE TABELAS PARA RELATÓRIO

print()

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

print('\033[1mQuartis de velocidade máxima\033[0m')
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

desvioPadraoVelocidade = dadosAustralia['Velocidade máxima'].std()
varianciaVelocidade = dadosAustralia['Velocidade máxima'].var()
coeficienteVariacaoVelocidade = (desvioPadraoVelocidade / mediaVelocidade) * 100

desvioPadraoHorarioMinutos = dadosAustralia['Horário'].std()
varianciaHorario = dadosAustralia['Horário'].var()
coeficienteVariacaoHorario = (desvioPadraoHorarioMinutos/mediaHorarioMinutos) * 100

desvioPadraoHorario = str(int(desvioPadraoHorarioMinutos // 60)).rjust(2, '0') + \
    ':' + str(int(desvioPadraoHorarioMinutos % 60)).rjust(2, '0')

valoresDesvioPadrao = {'Horário':  [desvioPadraoHorario],
                       'Velocidade Máxima(km/h)': [desvioPadraoVelocidade]
                       }
valoresVarianca = {'Horário':  [varianciaHorario],
                   'Velocidade Máxima': [varianciaVelocidade]
                   }
valoresCoeficienteVariacao = {'Horário(%)':  [coeficienteVariacaoHorario],
                              'Velocidade Máxima(%)': [coeficienteVariacaoVelocidade]
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
plt.hist(dadosAustralia['Velocidade máxima'], rwidth=0.9, bins=12)
plt.title('Distribuição de acidentes fatais por velocidade máxima da rodovia')
plt.xlabel('Velocidade (em km/h)')
x = np.arange(0, 131, 10)
plt.xticks(x)
plt.ylabel('Acidentes fatais')
plt.savefig('histograma-velocidade-maxima.png')
plt.show()
plt.close()

# GERANDO HISTOGRAMAS HORÁRIO

'''plt.figure(figsize=(10,7), dpi= 80)
sns.distplot(dadosAustralia["Velocidade máxima"], color="dodgerblue", label="Compact",)
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
boxplot = dadosAustralia.boxplot(column="Velocidade máxima", by="Gênero")
boxplot.plot()
plt.title('Distribuição de acidentes fatais: velocidade máxima por gênero')
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

# print(stats.normaltest(dadosAustralia['Velocidade máxima']))
dadosAustralia['HorarioH'] = dadosAustralia['HorarioH'].astype(int)

amostra = dadosAustralia.sample(100)
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

plt.xlabel(" ($) Gasto em propaganda de TV")
plt.ylabel(" ($) Vendas")
plt.show()
plt.savefig('dispersao2')
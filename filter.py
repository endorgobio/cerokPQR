import pandas as pd

dfPQR = pd.read_excel(r'data/PQRs.xlsx')


dfPQR_filtered = dfPQR[['Número Radicado', 'Canal Origen PQR', 'Edad Solicitante',
                        'Genero Solicitante', 'Riesgos Marcados Afectado', 
                        'Contenido de la PQRS', 'Código Patología', 'Patología',
                        'Riesgo de vida']]
#dfPQR_filtered = dfPQR_filtered[dfPQR_filtered['Riesgo de vida'].notna()]
dfPQR_filtered.to_csv(r'data/PQRs_filtered.csv')

# Informe sobre calidad de bibliotecas públicas en Bogotá

Este repositorio consolida y analiza mediciones de **iluminación**, **ruido** y **conectividad WiFi** tomadas en bibliotecas públicas de Bogotá para construir un informe técnico reproducible.

## Estructura

- `Datos/`: archivos fuente (`.xlsx`) de recolección.
- `Ejemplos/`: notebooks y PDF de referencia metodológica.
- `scripts/analisis_bibliotecas.py`: pipeline de limpieza, análisis y exportación.
- `Informe/Analisis_integral_bibliotecas_bogota.ipynb`: notebook integral del análisis.
- `Informe/informe_calidad_bibliotecas.tex`: informe técnico en LaTeX.
- `Informe/figuras/` y `Informe/tablas/`: artefactos generados para el informe.

## Requisitos

- Python 3.10+ (probado con 3.12)
- `pdflatex`

## Ejecución

```bash
python3 -m venv .venv
.venv/bin/pip install pandas openpyxl matplotlib seaborn scipy jupyter nbformat nbconvert
.venv/bin/python scripts/analisis_bibliotecas.py
```

Opcionalmente, ejecutar el notebook:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace Informe/Analisis_integral_bibliotecas_bogota.ipynb
```

Compilar informe:

```bash
cd Informe
pdflatex -interaction=nonstopmode -output-directory build informe_calidad_bibliotecas.tex
pdflatex -interaction=nonstopmode -output-directory build informe_calidad_bibliotecas.tex
```

PDF final: `Informe/build/informe_calidad_bibliotecas.pdf`.

## Criterios de análisis

- Luz: umbral por tipo de espacio (500/300/200 lux).
- Ruido: criterio de confort de lectura (`<= 40 dBA`).
- WiFi: velocidad de bajada mediana (`>= 10 Mbps`).
- Índice compuesto: promedio de cumplimiento de los 3 criterios (0 a 100).

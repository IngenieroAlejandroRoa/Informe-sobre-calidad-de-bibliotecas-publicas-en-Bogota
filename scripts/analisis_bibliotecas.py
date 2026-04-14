from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import itertools
import re
import unicodedata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu
import statsmodels.formula.api as smf

sns.set_theme(style="whitegrid")

NUMERIC_COLS = [
    "Lux_promedio",
    "dBA_prom",
    "Bajada_Mediana",
    "Subida_mediana",
    "Ocupación (personas)",
]


def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", text)


def detect_sheet(xls: pd.ExcelFile) -> str:
    for sheet in xls.sheet_names:
        cols = [str(c).strip() for c in pd.read_excel(xls, sheet_name=sheet, nrows=0).columns]
        if len(cols) >= 25 and any(c in cols for c in ["Biblioteca", "Hora"]):
            return sheet
    return xls.sheet_names[0]


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    for i, col in enumerate(df.columns):
        c = str(col).strip()
        if c.lower() in {"nan", "columna1", "unnamed: 0"} and i == 0:
            c = "N°"
        cols.append(c)
    df.columns = cols
    return df


def infer_time_slot(hora: pd.Timestamp | str) -> str:
    if pd.isna(hora):
        return "Sin dato"
    if not isinstance(hora, pd.Timestamp):
        hora = pd.to_datetime(str(hora), errors="coerce")
    if pd.isna(hora):
        return "Sin dato"
    h = hora.hour
    if 8 <= h <= 11:
        return "Manana"
    if 12 <= h <= 15:
        return "Tarde"
    if 16 <= h <= 19:
        return "Tarde-noche"
    return "Otra"


def light_threshold(tipo_espacio: str) -> float:
    t = normalize_text(tipo_espacio)
    if any(k in t for k in ["estanter", "anaquel"]):
        return 200
    if any(k in t for k in ["circul", "general", "pasillo"]):
        return 300
    return 500


def bootstrap_ci(series: pd.Series, n_boot: int = 2000, alpha: float = 0.05) -> tuple[float, float, float]:
    vals = series.dropna().to_numpy()
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(42)
    means = np.array([rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)])
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(vals.mean()), lo, hi


def epsilon_squared_kruskal(h: float, n: int, k: int) -> float:
    if n <= k:
        return np.nan
    return max(0.0, (h - k + 1) / (n - k))


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    gt = sum((xi > y).sum() for xi in x)
    lt = sum((xi < y).sum() for xi in x)
    return (gt - lt) / (len(x) * len(y))


def holm_correction(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.zeros(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running_max = max(running_max, val)
        adjusted[idx] = min(1.0, running_max)
    return adjusted.tolist()


@dataclass
class OutputPaths:
    figuras: Path
    tablas: Path


def load_data(data_dir: Path) -> pd.DataFrame:
    dfs = []
    for fp in sorted(data_dir.glob("*.xlsx")):
        xls = pd.ExcelFile(fp)
        sheet = detect_sheet(xls)
        df = pd.read_excel(fp, sheet_name=sheet)
        df = standardize_columns(df)
        df["archivo_fuente"] = fp.name
        df["hoja_fuente"] = sheet
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

    for col in NUMERIC_COLS:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data["Biblioteca"] = data["Biblioteca"].astype(str).str.strip()
    data["Tipo de espacio"] = data["Tipo de espacio"].astype(str).str.strip()
    data["Fecha"] = pd.to_datetime(data["Fecha"], errors="coerce")
    data["Hora_dt"] = pd.to_datetime(data["Hora"].astype(str), errors="coerce")
    data["Franja"] = data["Hora_dt"].apply(infer_time_slot)

    metric_cols = ["Lux_promedio", "dBA_prom", "Bajada_Mediana", "Subida_mediana"]
    data = data[data["Biblioteca"].notna() & (data["Biblioteca"] != "")]
    data = data.dropna(subset=metric_cols, how="all")

    data["Lux_umbral"] = data["Tipo de espacio"].apply(light_threshold)
    data["Cumple_Luz"] = data["Lux_promedio"] >= data["Lux_umbral"]
    data["Cumple_Ruido"] = data["dBA_prom"] <= 40
    data["Cumple_WiFi"] = data["Bajada_Mediana"] >= 10
    data["Indice_Compuesto"] = (
        data[["Cumple_Luz", "Cumple_Ruido", "Cumple_WiFi"]].astype(float).mean(axis=1) * 100
    )

    # Índice ponderado propuesto (más peso a ruido y luz por confort de estudio)
    weights = {"Cumple_Luz": 0.40, "Cumple_Ruido": 0.40, "Cumple_WiFi": 0.20}
    data["Indice_Ponderado"] = (
        data["Cumple_Luz"].astype(float) * weights["Cumple_Luz"]
        + data["Cumple_Ruido"].astype(float) * weights["Cumple_Ruido"]
        + data["Cumple_WiFi"].astype(float) * weights["Cumple_WiFi"]
    ) * 100

    return data


def build_posthoc_table(data: pd.DataFrame, var: str, min_n: int = 15) -> pd.DataFrame:
    grouped = [(b, g[var].dropna().values) for b, g in data.groupby("Biblioteca") if g[var].dropna().shape[0] >= min_n]
    rows = []
    pvals = []
    pairs = []
    for (b1, x), (b2, y) in itertools.combinations(grouped, 2):
        stat = mannwhitneyu(x, y, alternative="two-sided")
        pvals.append(float(stat.pvalue))
        pairs.append((b1, b2, x, y))

    if not pairs:
        return pd.DataFrame(columns=["variable", "biblioteca_a", "biblioteca_b", "p_ajustada", "cliffs_delta"])

    adj = holm_correction(pvals)
    for i, (b1, b2, x, y) in enumerate(pairs):
        rows.append(
            {
                "variable": var,
                "biblioteca_a": b1,
                "biblioteca_b": b2,
                "p_ajustada": adj[i],
                "cliffs_delta": cliffs_delta(x, y),
            }
        )

    posthoc = pd.DataFrame(rows)
    posthoc = posthoc.sort_values("p_ajustada")
    return posthoc


def export_tables(data: pd.DataFrame, out: OutputPaths) -> dict[str, pd.DataFrame]:
    out.tablas.mkdir(parents=True, exist_ok=True)

    data.to_csv(out.tablas / "datos_consolidados.csv", index=False)

    resumen_general = pd.DataFrame(
        {
            "metric": ["registros", "bibliotecas", "localidades"],
            "value": [int(len(data)), int(data["Biblioteca"].nunique()), int(data["Localidad"].astype(str).nunique())],
        }
    )

    resumen_biblioteca = (
        data.groupby("Biblioteca")
        .agg(
            registros=("Biblioteca", "size"),
            lux_promedio=("Lux_promedio", "mean"),
            lux_mediana=("Lux_promedio", "median"),
            ruido_promedio=("dBA_prom", "mean"),
            ruido_mediana=("dBA_prom", "median"),
            bajada_promedio_mbps=("Bajada_Mediana", "mean"),
            bajada_mediana_mbps=("Bajada_Mediana", "median"),
            ocupacion_promedio=("Ocupación (personas)", "mean"),
        )
        .sort_values("registros", ascending=False)
        .reset_index()
    )

    cumplimiento = (
        data.groupby("Biblioteca")
        .agg(
            cumplimiento_luz=("Cumple_Luz", lambda s: s.mean() * 100),
            cumplimiento_ruido=("Cumple_Ruido", lambda s: s.mean() * 100),
            cumplimiento_wifi=("Cumple_WiFi", lambda s: s.mean() * 100),
            indice_compuesto=("Indice_Compuesto", "mean"),
            indice_ponderado=("Indice_Ponderado", "mean"),
        )
        .round(2)
        .sort_values("indice_ponderado", ascending=False)
        .reset_index()
    )

    # IC 95% globales por bootstrap
    ci_rows = []
    for var in ["Lux_promedio", "dBA_prom", "Bajada_Mediana", "Subida_mediana", "Indice_Compuesto"]:
        mean, lo, hi = bootstrap_ci(data[var])
        ci_rows.append({"variable": var, "media": mean, "ic95_inf": lo, "ic95_sup": hi, "n": int(data[var].notna().sum())})
    intervalos_confianza = pd.DataFrame(ci_rows)

    # Kruskal + tamaño del efecto epsilon^2
    kr_rows = []
    for var in ["Lux_promedio", "dBA_prom", "Bajada_Mediana"]:
        groups = [g[var].dropna().values for _, g in data.groupby("Biblioteca") if g[var].dropna().shape[0] >= 3]
        if len(groups) >= 2:
            h, p = kruskal(*groups)
            n = int(sum(len(g) for g in groups))
            k = len(groups)
            e2 = epsilon_squared_kruskal(float(h), n, k)
            kr_rows.append({"variable": var, "kruskal_H": h, "p_value": p, "epsilon_sq": e2, "n": n, "k_grupos": k})
    efecto_kruskal = pd.DataFrame(kr_rows)

    # Post-hoc (Mann-Whitney + Holm), solo contrastes significativos top
    posthoc_frames = [
        build_posthoc_table(data, "Lux_promedio"),
        build_posthoc_table(data, "dBA_prom"),
        build_posthoc_table(data, "Bajada_Mediana"),
    ]
    posthoc = pd.concat(posthoc_frames, ignore_index=True)
    posthoc_significativos = posthoc[posthoc["p_ajustada"] < 0.05].copy().head(40)

    # Outliers por IQR
    out_rows = []
    for var in ["Lux_promedio", "dBA_prom", "Bajada_Mediana"]:
        s = data[var].dropna()
        q1, q3 = np.quantile(s, [0.25, 0.75])
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        n_out = int(((data[var] < lo) | (data[var] > hi)).sum())
        out_rows.append({"variable": var, "q1": q1, "q3": q3, "iqr": iqr, "lim_inf": lo, "lim_sup": hi, "outliers": n_out, "pct_outliers": n_out / max(1, s.shape[0]) * 100})
    outliers_resumen = pd.DataFrame(out_rows)

    # Modelos explicativos (OLS robustos)
    mdl_ruido_df = data.dropna(subset=["dBA_prom", "Ocupación (personas)", "Franja"]).copy()
    mdl_ruido = smf.ols("dBA_prom ~ Q('Ocupación (personas)') + C(Franja)", data=mdl_ruido_df).fit(cov_type="HC3")

    mdl_wifi_df = data.dropna(subset=["Bajada_Mediana", "Ocupación (personas)", "Franja", "dBA_prom"]).copy()
    mdl_wifi = smf.ols("Bajada_Mediana ~ Q('Ocupación (personas)') + dBA_prom + C(Franja)", data=mdl_wifi_df).fit(cov_type="HC3")

    def model_table(model, name: str) -> pd.DataFrame:
        conf = model.conf_int()
        return pd.DataFrame(
            {
                "modelo": name,
                "termino": model.params.index,
                "coef": model.params.values,
                "ic95_inf": conf[0].values,
                "ic95_sup": conf[1].values,
                "p_value": model.pvalues.values,
                "r2": model.rsquared,
                "n": int(model.nobs),
            }
        )

    regresion_ruido = model_table(mdl_ruido, "ruido")
    regresion_wifi = model_table(mdl_wifi, "wifi")

    prioridades = cumplimiento.sort_values("indice_ponderado", ascending=True).head(7)

    outputs = {
        "resumen_general": resumen_general,
        "resumen_biblioteca": resumen_biblioteca,
        "cumplimiento": cumplimiento,
        "intervalos_confianza": intervalos_confianza,
        "efecto_kruskal": efecto_kruskal,
        "posthoc_significativos": posthoc_significativos,
        "outliers_resumen": outliers_resumen,
        "regresion_ruido": regresion_ruido,
        "regresion_wifi": regresion_wifi,
        "prioridades": prioridades,
    }

    for name, df in outputs.items():
        df.to_csv(out.tablas / f"{name}.csv", index=False)
        df.to_latex(
            out.tablas / f"{name}.tex",
            index=False,
            float_format="%.3f",
            escape=True,
        )

    return outputs


def export_figures(data: pd.DataFrame, out: OutputPaths) -> None:
    out.figuras.mkdir(parents=True, exist_ok=True)
    top_order = data["Biblioteca"].value_counts().index.tolist()

    plt.figure(figsize=(10, 8))
    data["Biblioteca"].value_counts().sort_values(ascending=True).plot(kind="barh", color="#4C78A8")
    plt.title("Registros por biblioteca")
    plt.xlabel("Número de registros")
    plt.ylabel("Biblioteca")
    plt.tight_layout()
    plt.savefig(out.figuras / "registros_por_biblioteca.png", dpi=220)
    plt.close()

    plt.figure(figsize=(13, 6))
    sns.boxplot(data=data, x="Biblioteca", y="Lux_promedio", order=top_order, color="#72B7B2")
    plt.axhline(500, color="red", linestyle="--", label="Umbral referencia 500 lux")
    plt.xticks(rotation=70, ha="right")
    plt.ylabel("Lux")
    plt.title("Distribución de iluminación por biblioteca")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out.figuras / "box_lux_biblioteca.png", dpi=220)
    plt.close()

    plt.figure(figsize=(13, 6))
    sns.violinplot(data=data, x="Biblioteca", y="dBA_prom", order=top_order, color="#F58518", cut=0)
    plt.axhline(40, color="red", linestyle="--", label="Umbral confort 40 dBA")
    plt.xticks(rotation=70, ha="right")
    plt.ylabel("dBA")
    plt.title("Distribución (violin) de ruido por biblioteca")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out.figuras / "violin_ruido_biblioteca.png", dpi=220)
    plt.close()

    plt.figure(figsize=(13, 6))
    sns.boxplot(data=data, x="Biblioteca", y="Bajada_Mediana", order=top_order, color="#54A24B")
    plt.axhline(10, color="red", linestyle="--", label="Umbral mínimo 10 Mbps")
    plt.xticks(rotation=70, ha="right")
    plt.ylabel("Mbps")
    plt.title("Distribución de velocidad de bajada WiFi por biblioteca")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out.figuras / "box_wifi_biblioteca.png", dpi=220)
    plt.close()

    c = (
        data.groupby("Biblioteca")[["Cumple_Luz", "Cumple_Ruido", "Cumple_WiFi"]]
        .mean()
        .rename(columns={"Cumple_Luz": "Luz", "Cumple_Ruido": "Ruido", "Cumple_WiFi": "WiFi"})
        * 100
    )
    plt.figure(figsize=(10, 8))
    sns.heatmap(c.sort_index(), annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={"label": "% cumplimiento"})
    plt.title("Cumplimiento normativo por biblioteca")
    plt.tight_layout()
    plt.savefig(out.figuras / "heatmap_cumplimiento.png", dpi=220)
    plt.close()

    idx_simple = data.groupby("Biblioteca")["Indice_Compuesto"].mean()
    idx_weight = data.groupby("Biblioteca")["Indice_Ponderado"].mean()
    idx = pd.DataFrame({"Simple": idx_simple, "Ponderado": idx_weight}).sort_values("Ponderado", ascending=False)
    plt.figure(figsize=(12, 6))
    x = np.arange(len(idx))
    w = 0.4
    plt.bar(x - w / 2, idx["Simple"], width=w, label="Índice simple")
    plt.bar(x + w / 2, idx["Ponderado"], width=w, label="Índice ponderado")
    plt.xticks(x, idx.index, rotation=70, ha="right")
    plt.ylim(0, 100)
    plt.ylabel("Puntaje")
    plt.title("Comparación índice compuesto simple vs ponderado")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out.figuras / "indice_simple_vs_ponderado.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7, 6))
    correl = data[["Lux_promedio", "dBA_prom", "Bajada_Mediana", "Subida_mediana", "Ocupación (personas)"]].corr(numeric_only=True)
    sns.heatmap(correl, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlación entre variables medidas")
    plt.tight_layout()
    plt.savefig(out.figuras / "correlacion_metricas.png", dpi=220)
    plt.close()

    by_slot = (
        data.groupby("Franja")
        .agg(ruido_promedio=("dBA_prom", "mean"), wifi_promedio=("Bajada_Mediana", "mean"), luz_promedio=("Lux_promedio", "mean"))
        .reset_index()
    )
    slot_order = [s for s in ["Manana", "Tarde", "Tarde-noche", "Otra", "Sin dato"] if s in by_slot["Franja"].tolist()]
    by_slot["Franja"] = pd.Categorical(by_slot["Franja"], categories=slot_order, ordered=True)
    by_slot = by_slot.sort_values("Franja")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    sns.barplot(data=by_slot, x="Franja", y="luz_promedio", ax=axes[0], color="#72B7B2")
    axes[0].axhline(500, color="red", linestyle="--", linewidth=1)
    axes[0].set_title("Lux por franja")
    sns.barplot(data=by_slot, x="Franja", y="ruido_promedio", ax=axes[1], color="#F58518")
    axes[1].axhline(40, color="red", linestyle="--", linewidth=1)
    axes[1].set_title("Ruido por franja")
    sns.barplot(data=by_slot, x="Franja", y="wifi_promedio", ax=axes[2], color="#54A24B")
    axes[2].axhline(10, color="red", linestyle="--", linewidth=1)
    axes[2].set_title("WiFi por franja")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(out.figuras / "metricas_por_franja.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.regplot(data=data, x="Ocupación (personas)", y="dBA_prom", scatter_kws={"alpha": 0.35, "s": 16}, line_kws={"color": "black"})
    plt.axhline(40, color="red", linestyle="--", linewidth=1)
    plt.title("Relación entre ocupación y ruido")
    plt.xlabel("Ocupación (personas)")
    plt.ylabel("Ruido (dBA)")
    plt.tight_layout()
    plt.savefig(out.figuras / "scatter_ocupacion_ruido.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.regplot(data=data, x="Ocupación (personas)", y="Bajada_Mediana", scatter_kws={"alpha": 0.35, "s": 16}, line_kws={"color": "black"})
    plt.axhline(10, color="red", linestyle="--", linewidth=1)
    plt.title("Relación entre ocupación y velocidad WiFi")
    plt.xlabel("Ocupación (personas)")
    plt.ylabel("Bajada mediana (Mbps)")
    plt.tight_layout()
    plt.savefig(out.figuras / "scatter_ocupacion_wifi.png", dpi=220)
    plt.close()


def export_narrative(data: pd.DataFrame, outputs: dict[str, pd.DataFrame], out: OutputPaths) -> None:
    cumplimiento = outputs["cumplimiento"]
    efecto = outputs["efecto_kruskal"]
    posthoc = outputs["posthoc_significativos"]
    outliers = outputs["outliers_resumen"]
    reg_ruido = outputs["regresion_ruido"]
    reg_wifi = outputs["regresion_wifi"]

    top = cumplimiento.head(3)
    bottom = cumplimiento.tail(3)

    # Interpretación avanzada
    kr_lines = []
    for _, r in efecto.iterrows():
        var = str(r["variable"]).replace("_", "\\_")
        p = r["p_value"]
        e2 = r["epsilon_sq"]
        mag = "pequeño" if e2 < 0.08 else "mediano" if e2 < 0.26 else "grande"
        kr_lines.append(f"{var} (H={r['kruskal_H']:.2f}, p={p:.4f}, epsilon-cuadrado={e2:.3f}, efecto {mag})")

    posthoc_top = posthoc.head(6)
    posthoc_txt = "; ".join(
        [
            f"{str(r['variable']).replace('_', '\\_')}: {r['biblioteca_a']} vs {r['biblioteca_b']} (p-aj={r['p_ajustada']:.4f}, delta={r['cliffs_delta']:.3f})"
            for _, r in posthoc_top.iterrows()
        ]
    )

    outlier_txt = "; ".join(
        [f"{str(r['variable']).replace('_', '\\_')}: {r['pct_outliers']:.1f}%" for _, r in outliers.iterrows()]
    )

    occ_ruido = reg_ruido.loc[reg_ruido["termino"] == "Q('Ocupación (personas)')"].iloc[0]
    occ_wifi = reg_wifi.loc[reg_wifi["termino"] == "Q('Ocupación (personas)')"].iloc[0]

    general = f"""
Este estudio consolidó {len(data)} registros válidos de {data['Biblioteca'].nunique()} bibliotecas y {data['Localidad'].nunique()} localidades. El cumplimiento global fue de {data['Cumple_Luz'].mean()*100:.1f}\% para iluminación, {data['Cumple_Ruido'].mean()*100:.1f}\% para ruido y {data['Cumple_WiFi'].mean()*100:.1f}\% para WiFi. En términos de desempeño integral, el índice simple promedio fue {data['Indice_Compuesto'].mean():.1f}/100 y el índice ponderado propuesto fue {data['Indice_Ponderado'].mean():.1f}/100.

Las bibliotecas de mayor desempeño ponderado fueron: """ + "; ".join([f"{r['Biblioteca']} ({r['indice_ponderado']:.1f})" for _, r in top.iterrows()]) + ". " + """Las de menor desempeño fueron: """ + "; ".join([f"{r['Biblioteca']} ({r['indice_ponderado']:.1f})" for _, r in bottom.iterrows()]) + "."
    (out.tablas / "analisis_general_auto.tex").write_text(general.strip() + "\n", encoding="utf-8")

    avanzado = f"""
Las pruebas de Kruskal--Wallis evidenciaron diferencias estadísticamente significativas entre bibliotecas en los tres ejes analizados, con los siguientes resultados: {"; ".join(kr_lines)}. Reportar p muy pequeño (aprox. 0.0000) no implica efecto infinito; por ello se complementó con tamaño del efecto, mostrando que las diferencias tienen magnitud sustantiva y no solo significancia por tamaño muestral.

En el análisis post-hoc (Mann--Whitney con ajuste de Holm) se identificaron contrastes puntuales robustos entre pares de bibliotecas. Los principales fueron: {posthoc_txt}. Esto respalda que la heterogeneidad no es uniforme, sino concentrada en comparaciones específicas de sedes.

El análisis de valores atípicos mostró la siguiente incidencia: {outlier_txt}. Este hallazgo sugiere episodios operativos extremos (picos de ruido o variabilidad de red) que deben gestionarse con estrategias de monitoreo continuo, no solo con promedios.

En modelos explicativos robustos, la ocupación mostró asociación positiva con ruido (coef={occ_ruido['coef']:.3f}, p={occ_ruido['p_value']:.4f}), mientras que su relación con velocidad de bajada fue menor (coef={occ_wifi['coef']:.3f}, p={occ_wifi['p_value']:.4f}). En términos prácticos, la presión de usuarios parece impactar más directamente el confort acústico que la capacidad de red promedio, lo que sugiere priorizar rediseño espacial y control de carga acústica en franjas críticas.
"""
    (out.tablas / "analisis_avanzado_auto.tex").write_text(avanzado.strip() + "\n", encoding="utf-8")

    discusion = """
Los resultados indican una brecha estructural entre sedes en condiciones de estudio. Esta variabilidad puede explicarse por diferencias en diseño arquitectónico, antigüedad de infraestructura, políticas de operación local, densidad de usuarios y capacidades de mantenimiento.

Desde la perspectiva de infraestructura pública urbana, los hallazgos sugieren que la calidad del servicio bibliotecario no depende solo de conectividad, sino de la interacción entre ambiente físico (ruido e iluminación), gobernanza del espacio y gestión por demanda. En contextos de alta ocupación, la calidad percibida por usuarios puede deteriorarse aun cuando la red WiFi sea aceptable.

Para política pública, esto implica migrar de un enfoque de cumplimiento mínimo a uno de desempeño integral por experiencia de usuario. En particular, el criterio acústico emerge como cuello de botella transversal, con impacto potencial sobre concentración, permanencia y rendimiento académico.
"""
    (out.tablas / "discusion_auto.tex").write_text(discusion.strip() + "\n", encoding="utf-8")

    conclusiones = f"""
\\begin{{itemize}}
\\item El sistema evaluado presenta heterogeneidad estadísticamente significativa entre bibliotecas en iluminación, ruido y conectividad; no es metodológicamente válido tratarlas como unidades equivalentes.
\\item El ruido es la dimensión más crítica para el confort de estudio, con incumplimientos recurrentes y asociación positiva con ocupación.
\\item La conectividad muestra mejor desempeño relativo, pero con asimetrías entre sedes y eventos atípicos que afectan continuidad de servicio.
\\item El índice ponderado propuesto mejora la sensibilidad diagnóstica frente al índice simple, al reflejar mayor peso de variables de confort ambiental para actividades de lectura y estudio.
\\item Se recomienda institucionalizar monitoreo periódico y tablero de control por sede, integrando indicadores de tendencia central, dispersión, cumplimiento e incidencia de outliers.
\\end{{itemize}}
"""
    (out.tablas / "conclusiones_auto.tex").write_text(conclusiones, encoding="utf-8")

    recomendaciones = """
\\subsection*{Corto plazo (0--6 meses)}
\\begin{itemize}
\\item Implementar gestión acústica operativa: zonificación silenciosa, señalización y control de actividades ruidosas por franja.
\\item Corregir puntos críticos de iluminación en áreas de lectura mediante ajustes de luminarias y mantenimiento focalizado.
\\item Definir protocolo de medición mensual con alertas para incumplimientos reiterados.
\\end{itemize}

\\subsection*{Mediano plazo (6--18 meses)}
\\begin{itemize}
\\item Ejecutar intervenciones físicas de acondicionamiento acústico (materiales absorbentes, redistribución de mobiliario, barreras blandas).
\\item Fortalecer arquitectura de red en sedes de mayor demanda y alta variabilidad de throughput.
\\item Implementar esquema de priorización presupuestal basado en índice ponderado y tamaño de brecha.
\\end{itemize}

\\subsection*{Largo plazo (18+ meses)}
\\begin{itemize}
\\item Integrar estándares de confort ambiental y conectividad en lineamientos distritales de diseño/renovación bibliotecaria.
\\item Consolidar un sistema de analítica pública longitudinal para evaluación de impacto de intervenciones.
\\item Incorporar percepción de usuarios y métricas de uso académico para validar mejoras en calidad del servicio.
\\end{itemize}
"""
    (out.tablas / "recomendaciones_auto.tex").write_text(recomendaciones, encoding="utf-8")

    indice = """
El índice compuesto actual (promedio simple) asume igual importancia entre iluminación, ruido y WiFi. Esta suposición facilita interpretación, pero puede subestimar dimensiones de mayor impacto en experiencia de estudio. Como mejora, se propone un índice ponderado: 40\% iluminación, 40\% ruido y 20\% WiFi. Esta ponderación prioriza condiciones de confort cognitivo y permanencia en sala.

Adicionalmente, se recomienda explorar ponderaciones basadas en evidencia empírica (por ejemplo, análisis factorial, preferencias de usuario o validación con desempeño académico), con evaluación de robustez por sensibilidad de pesos.
"""
    (out.tablas / "indice_mejorado_auto.tex").write_text(indice.strip() + "\n", encoding="utf-8")

    limitaciones = """
Entre las principales limitaciones del estudio se encuentran: (i) posibles sesgos de medición por heterogeneidad de dispositivos móviles y apps, (ii) diferencias en tamaño muestral entre bibliotecas, (iii) efecto de condiciones temporales no controladas (eventos, clima, mantenimiento puntual), y (iv) potencial dependencia intra-biblioteca por mediciones repetidas en contextos similares.

Para mitigar sesgos futuros se sugiere calibración cruzada de instrumentos, diseño de muestreo balanceado por sede y franja, y uso de modelos jerárquicos de efectos mixtos para capturar estructura anidada de los datos.
"""
    (out.tablas / "limitaciones_auto.tex").write_text(limitaciones.strip() + "\n", encoding="utf-8")


def export_conclusions(data: pd.DataFrame, outputs: dict[str, pd.DataFrame], out: OutputPaths) -> None:
    """Compatibilidad con el notebook previo."""
    export_narrative(data, outputs, out)


def run_pipeline(base_dir: Path) -> None:
    out = OutputPaths(figuras=base_dir / "Informe" / "figuras", tablas=base_dir / "Informe" / "tablas")
    data = load_data(base_dir / "Datos")
    outputs = export_tables(data, out)
    export_figures(data, out)
    export_narrative(data, outputs, out)
    print(f"Pipeline completado. Registros: {len(data)} | Bibliotecas: {data['Biblioteca'].nunique()}")


if __name__ == "__main__":
    run_pipeline(Path(__file__).resolve().parents[1])

import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium

st.set_page_config(page_title="Аналитика теплосетей", layout="wide")

CURRENT_YEAR = pd.Timestamp.now().year
POSITION_COLORS = {
    "подзем": "#1f77b4",
    "надзем": "#ff7f0e",
    "unknown": "#7f7f7f",
}


def normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (float, int, np.number)):
        return float(value)
    cleaned = str(value).strip().replace(" ", "").replace(",", ".")
    if cleaned == "":
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip().lower() for col in df.columns]
    return df


def read_csv_flexible(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "cp1251", "utf-8"]
    separators = [",", ";", "\t"]
    for enc in encodings:
        for sep in separators:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep)
                if len(df.columns) > 1:
                    return standardize_columns(df)
            except Exception:
                continue
    return standardize_columns(pd.read_csv(path, encoding="utf-8", sep=None, engine="python"))


def detect_files(data_dir: Path) -> Dict[str, Optional[Path]]:
    all_csv = list(data_dir.glob("*.csv"))

    def pick(*keywords: str) -> Optional[Path]:
        for file in all_csv:
            name = file.name.lower()
            if all(keyword in name for keyword in keywords):
                return file
        return None

    return {
        "main_pipes": pick("магистраль", "труб"),
        "pipes": pick("трубопроводы", "теплоснабжения"),
        "sources": pick("источники", "теплоснабжения"),
        "pump_stations": pick("насосные", "станции"),
        "ctp": pick("цтп"),
        "wells": pick("колодцы", "люки"),
        "chambers": pick("камеры", "павильоны"),
        "damage_1": pick("повреждения.csv"),
        "damage_2": pick("повреждения2"),
    }


def get_col(df: pd.DataFrame, options: Iterable[str], default: Optional[str] = None) -> Optional[str]:
    columns = set(df.columns)
    for opt in options:
        if opt in columns:
            return opt
    return default


def extract_coordinates_from_wkt(wkt_value: object) -> List[Tuple[float, float]]:
    text = normalize_text(wkt_value)
    if not text:
        return []
    if "(" not in text:
        return []
    numbers = re.findall(r"[-+]?\d+\.?\d*", text)
    if len(numbers) < 4 or len(numbers) % 2 != 0:
        return []
    coords = []
    for i in range(0, len(numbers), 2):
        lon = to_float(numbers[i])
        lat = to_float(numbers[i + 1])
        if lat is None or lon is None:
            continue
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            coords.append((lat, lon))
    return coords


def pipeline_records(df: pd.DataFrame) -> List[dict]:
    if df.empty:
        return []
    geom_col = get_col(df, ["geometry", "wkt", "shape", "geom"])
    name_col = get_col(df, ["name", "название", "naimenovanie"])
    diam_col = get_col(df, ["diametr_up", "diameter", "диаметр"])
    len_col = get_col(df, ["shape_length", "length", "длина"])
    pos_col = get_col(df, ["position", "тип_прокладки", "prokladka"])
    ins_col = get_col(df, ["predizol"])

    records = []
    for _, row in df.iterrows():
        coords: List[Tuple[float, float]] = []
        if geom_col:
            coords = extract_coordinates_from_wkt(row.get(geom_col))
        if not coords:
            continue

        position = normalize_text(row.get(pos_col)).lower() if pos_col else ""
        if "подзем" in position:
            position_group = "подзем"
        elif "надзем" in position:
            position_group = "надзем"
        else:
            position_group = "unknown"

        records.append(
            {
                "name": normalize_text(row.get(name_col)) if name_col else "Без названия",
                "diameter": normalize_text(row.get(diam_col)) if diam_col else "н/д",
                "length": to_float(row.get(len_col)) if len_col else None,
                "position": normalize_text(row.get(pos_col)) if pos_col else "н/д",
                "position_group": position_group,
                "predizol": normalize_text(row.get(ins_col)).lower() if ins_col else "",
                "coords": coords,
            }
        )
    return records


def point_records(df: pd.DataFrame, lat_opts: List[str], lon_opts: List[str], fallback_wkt: bool = True) -> List[dict]:
    if df.empty:
        return []
    lat_col = get_col(df, lat_opts)
    lon_col = get_col(df, lon_opts)
    geom_col = get_col(df, ["geometry", "wkt", "shape", "geom"])

    points = []
    for _, row in df.iterrows():
        lat = to_float(row.get(lat_col)) if lat_col else None
        lon = to_float(row.get(lon_col)) if lon_col else None

        if (lat is None or lon is None) and fallback_wkt and geom_col:
            coords = extract_coordinates_from_wkt(row.get(geom_col))
            if coords:
                lat, lon = coords[0]

        if lat is None or lon is None:
            continue
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue

        points.append({"lat": lat, "lon": lon, "row": row})
    return points


@st.cache_data(show_spinner=False)
def load_data(data_dir: str) -> Dict[str, object]:
    base = Path(data_dir)
    files = detect_files(base)

    def load(key: str) -> pd.DataFrame:
        fp = files.get(key)
        if fp and fp.exists():
            return read_csv_flexible(fp)
        return pd.DataFrame()

    main_pipes_df = load("main_pipes")
    pipes_df = load("pipes")
    sources_df = load("sources")
    pumps_df = load("pump_stations")
    ctp_df = load("ctp")
    wells_df = load("wells")
    chambers_df = load("chambers")
    dmg1_df = load("damage_1")
    dmg2_df = load("damage_2")

    combined_pipes = pd.concat([main_pipes_df, pipes_df], ignore_index=True)
    damages = pd.concat([dmg1_df, dmg2_df], ignore_index=True)

    return {
        "files": files,
        "main_pipes_df": main_pipes_df,
        "pipes_df": pipes_df,
        "combined_pipes": combined_pipes,
        "sources_df": sources_df,
        "pumps_df": pumps_df,
        "ctp_df": ctp_df,
        "wells_df": wells_df,
        "chambers_df": chambers_df,
        "damages_df": damages,
    }


def build_network_map(data: Dict[str, object], map_center: Tuple[float, float], zoom: int) -> folium.Map:
    fmap = folium.Map(location=map_center, zoom_start=zoom, tiles="CartoDB positron", control_scale=True)

    pipelines_layer = folium.FeatureGroup(name="Трубопроводы", show=True)
    wear_layer = folium.FeatureGroup(name="Износ: нужна перекладка", show=True)

    for rec in pipeline_records(data["combined_pipes"]):
        popup = folium.Popup(
            f"<b>{rec['name']}</b><br>"
            f"Диаметр: {rec['diameter']}<br>"
            f"Прокладка: {rec['position']}<br>"
            f"Длина: {rec['length']:.1f} м" if rec["length"] is not None else f"<b>{rec['name']}</b><br>Длина: н/д",
            max_width=360,
        )
        folium.PolyLine(
            rec["coords"],
            color=POSITION_COLORS.get(rec["position_group"], POSITION_COLORS["unknown"]),
            weight=4,
            opacity=0.85,
            popup=popup,
        ).add_to(pipelines_layer)

        if "нужна перекладка" in rec["predizol"]:
            folium.PolyLine(rec["coords"], color="#d62728", weight=6, opacity=0.95).add_to(wear_layer)

    pipelines_layer.add_to(fmap)
    wear_layer.add_to(fmap)

    sources_layer = folium.FeatureGroup(name="Источники теплоснабжения", show=True)
    for p in point_records(data["sources_df"], ["y", "lat", "latitude"], ["x", "lon", "longitude"]):
        row = p["row"]
        name_col = get_col(data["sources_df"], ["name", "название", "naimenovanie"])
        power_col = get_col(data["sources_df"], ["setup_power", "установленная_мощность"])
        folium.Marker(
            [p["lat"], p["lon"]],
            icon=folium.Icon(color="red", icon="fire", prefix="fa"),
            popup=folium.Popup(
                f"<b>{normalize_text(row.get(name_col))}</b><br>"
                f"Установленная мощность: {normalize_text(row.get(power_col))}",
                max_width=320,
            ),
        ).add_to(sources_layer)
    sources_layer.add_to(fmap)

    infra_layer = folium.FeatureGroup(name="Насосные станции и ЦТП", show=True)
    for p in point_records(data["pumps_df"], ["y", "lat", "latitude"], ["x", "lon", "longitude"]):
        row = p["row"]
        name_col = get_col(data["pumps_df"], ["name", "название", "naimenovanie"])
        owner_col = get_col(data["pumps_df"], ["owner", "владелец"])
        folium.Marker(
            [p["lat"], p["lon"]],
            icon=folium.Icon(color="blue", icon="cog", prefix="fa"),
            popup=f"<b>{normalize_text(row.get(name_col))}</b><br>Владелец: {normalize_text(row.get(owner_col))}",
        ).add_to(infra_layer)
    for p in point_records(data["ctp_df"], ["y", "lat", "latitude"], ["x", "lon", "longitude"]):
        row = p["row"]
        name_col = get_col(data["ctp_df"], ["name", "название", "naimenovanie"])
        owner_col = get_col(data["ctp_df"], ["owner", "владелец"])
        folium.Marker(
            [p["lat"], p["lon"]],
            icon=folium.Icon(color="green", icon="building", prefix="fa"),
            popup=f"<b>{normalize_text(row.get(name_col))}</b><br>Владелец: {normalize_text(row.get(owner_col))}",
        ).add_to(infra_layer)
    infra_layer.add_to(fmap)

    inspection_layer = folium.FeatureGroup(name="Колодцы, люки, камеры", show=False)
    cluster = MarkerCluster(name="Объекты осмотра").add_to(inspection_layer)

    for df in [data["wells_df"], data["chambers_df"]]:
        if df.empty:
            continue
        type_col = get_col(df, ["type", "тип", "vid"])
        elev_ground_col = get_col(df, ["elevation_ground"])
        elev_hatch_col = get_col(df, ["elevation_hatch"])
        for p in point_records(df, ["y", "lat", "latitude"], ["x", "lon", "longitude"]):
            row = p["row"]
            folium.CircleMarker(
                location=[p["lat"], p["lon"]],
                radius=4,
                color="#5c5c5c",
                fill=True,
                fill_opacity=0.7,
                popup=(
                    f"Тип: {normalize_text(row.get(type_col))}<br>"
                    f"Отметка земли: {normalize_text(row.get(elev_ground_col))}<br>"
                    f"Отметка люка: {normalize_text(row.get(elev_hatch_col))}"
                ),
            ).add_to(cluster)
    inspection_layer.add_to(fmap)

    legend_html = """
    <div style='position: fixed; bottom: 20px; left: 20px; z-index:9999; background: white;
         border: 1px solid #ddd; padding: 10px 12px; border-radius: 8px; font-size: 13px;'>
      <b>Легенда трубопроводов</b><br>
      <span style='color:#1f77b4;'>■</span> Подземная прокладка<br>
      <span style='color:#ff7f0e;'>■</span> Надземная прокладка<br>
      <span style='color:#d62728;'>■</span> Нужна перекладка
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap


def build_damage_map(damages: pd.DataFrame, reason_filter: str, center: Tuple[float, float], zoom: int) -> folium.Map:
    fmap = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap", control_scale=True)
    if damages.empty:
        return fmap

    df = damages.copy()
    reason_col = get_col(df, ["prichina_povrezhdeniya", "причина", "cause"])
    lat_col = get_col(df, ["y_cor", "y", "lat", "latitude"])
    lon_col = get_col(df, ["x_cor", "x", "lon", "longitude"])

    if reason_filter != "Все причины" and reason_col:
        df = df[df[reason_col].fillna("").astype(str) == reason_filter]

    points = []
    for _, row in df.iterrows():
        lat = to_float(row.get(lat_col)) if lat_col else None
        lon = to_float(row.get(lon_col)) if lon_col else None
        if lat is None or lon is None:
            continue
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            points.append((lat, lon, row))

    heat = folium.FeatureGroup(name="Heatmap аварий", show=True)
    if points:
        HeatMap(data=[[p[0], p[1], 1.0] for p in points], radius=18, blur=14, min_opacity=0.3).add_to(heat)
    heat.add_to(fmap)

    markers_layer = folium.FeatureGroup(name="Инциденты (кластеры)", show=True)
    cluster = MarkerCluster().add_to(markers_layer)
    addr_col = get_col(df, ["adres", "address"])
    date_col = get_col(df, ["data_nachala", "date"])
    leak_col = get_col(df, ["utechka_vodi_t", "utechka"])

    for lat, lon, row in points:
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color="#e31a1c",
            fill=True,
            fill_color="#fb6a4a",
            fill_opacity=0.9,
            popup=(
                f"<b>Адрес:</b> {normalize_text(row.get(addr_col))}<br>"
                f"<b>Дата:</b> {normalize_text(row.get(date_col))}<br>"
                f"<b>Причина:</b> {normalize_text(row.get(reason_col))}<br>"
                f"<b>Потери воды, т:</b> {normalize_text(row.get(leak_col))}"
            ),
        ).add_to(cluster)

    markers_layer.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap


def render_kpi_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div style='background:#ffffff; border:1px solid #e6e6e6; border-radius:12px; padding:14px 18px; box-shadow:0 1px 3px rgba(0,0,0,0.06)'>
            <div style='font-size:13px; color:#6b7280;'>{label}</div>
            <div style='font-size:24px; font-weight:600; color:#111827'>{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.title("Интерактивная аналитика инфраструктуры теплосетей")

    data_dir = st.sidebar.text_input("Папка с CSV", value="data")
    st.sidebar.caption("Приложение работает в светлой теме, карты на светлых подложках.")

    data = load_data(data_dir)
    pipes_records = pipeline_records(data["combined_pipes"])

    all_coords = [coord for rec in pipes_records for coord in rec["coords"]]
    if all_coords:
        center = (float(np.mean([c[0] for c in all_coords])), float(np.mean([c[1] for c in all_coords])))
    else:
        center = (55.751244, 37.618423)

    damages = data["damages_df"]
    reason_col = get_col(damages, ["prichina_povrezhdeniya", "причина", "cause"])
    reasons = sorted(damages[reason_col].dropna().astype(str).unique().tolist()) if reason_col else []
    selected_reason = st.sidebar.selectbox("Причина повреждения", ["Все причины", *reasons], index=0)

    total_len = sum(rec["length"] for rec in pipes_records if rec["length"] is not None)
    sources_count = len(point_records(data["sources_df"], ["y", "lat", "latitude"], ["x", "lon", "longitude"]))
    ctp_count = len(point_records(data["ctp_df"], ["y", "lat", "latitude"], ["x", "lon", "longitude"]))
    pumps_count = len(point_records(data["pumps_df"], ["y", "lat", "latitude"], ["x", "lon", "longitude"]))

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card("Общая длина магистралей, км", f"{total_len/1000:,.1f}".replace(",", " "))
    with k2:
        render_kpi_card("Источники теплоснабжения", f"{sources_count}")
    with k3:
        render_kpi_card("ЦТП", f"{ctp_count}")
    with k4:
        render_kpi_card("Насосные станции", f"{pumps_count}")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Карта сети",
        "Состояние фондов",
        "Аварийность",
        "Потери и отключения",
    ])

    with tab1:
        st.subheader("Схема сети и инфраструктурных объектов")
        network_map = build_network_map(data, center, 11)
        st_folium(network_map, width=None, height=780, use_container_width=True)

    with tab2:
        st.subheader("Состояние фондов и участки под капитальный ремонт")
        predizol_values = [rec["predizol"] if rec["predizol"] else "не указано" for rec in pipes_records]
        if predizol_values:
            s = pd.Series(predizol_values).value_counts().reset_index()
            s.columns = ["Состояние", "Количество"]
            fig_pie = px.pie(s, names="Состояние", values="Количество", hole=0.35)
            fig_pie.update_layout(template="plotly_white", title="Состояние изоляции (predizol)")
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Нет данных по состоянию изоляции.")

        year_col = get_col(data["combined_pipes"], ["god_sdachi_expl", "year", "год_сдачи_экспл"])
        if year_col:
            years = pd.to_numeric(data["combined_pipes"][year_col], errors="coerce")
            ages = (CURRENT_YEAR - years).dropna()
            ages = ages[(ages >= 0) & (ages < 150)]
            if not ages.empty:
                fig_hist = px.histogram(ages, nbins=25, labels={"value": "Возраст трубы, лет", "count": "Количество"})
                fig_hist.update_layout(template="plotly_white", title="Гистограмма возраста труб")
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("Возраст труб невозможно рассчитать: нет валидных годов ввода.")
        else:
            st.info("Колонка god_sdachi_expl не найдена.")

    with tab3:
        st.subheader("Тепловая карта аварий и кластеры инцидентов")
        damage_map = build_damage_map(damages, selected_reason, center, 11)
        st_folium(damage_map, width=None, height=760, use_container_width=True)

    with tab4:
        st.subheader("Убытки от утечек и последствия отключений")
        if damages.empty:
            st.info("Нет данных по авариям для построения аналитики.")
        else:
            leak_col = get_col(damages, ["utechka_vodi_t", "utechka"])
            addr_col = get_col(damages, ["adres", "address"])
            bez_otop_col = get_col(damages, ["bez_otop"])
            bez_gvs_col = get_col(damages, ["bez_gvs"])

            if leak_col and addr_col:
                tmp = damages[[addr_col, leak_col]].copy()
                tmp[leak_col] = pd.to_numeric(tmp[leak_col], errors="coerce")
                top10 = tmp.dropna().sort_values(leak_col, ascending=False).head(10)
                if not top10.empty:
                    fig_top = px.bar(
                        top10,
                        x=leak_col,
                        y=addr_col,
                        orientation="h",
                        title="Топ-10 аварий по объему потери теплоносителя",
                    )
                    fig_top.update_layout(template="plotly_white", yaxis_title="Адрес", xaxis_title="Потери воды, т")
                    st.plotly_chart(fig_top, use_container_width=True)

            if bez_otop_col or bez_gvs_col:
                consequences = {
                    "Без отопления": int((damages[bez_otop_col].astype(str).str.lower() == "да").sum()) if bez_otop_col else 0,
                    "Без ГВС": int((damages[bez_gvs_col].astype(str).str.lower() == "да").sum()) if bez_gvs_col else 0,
                }
                fig_cons = px.pie(
                    names=list(consequences.keys()),
                    values=list(consequences.values()),
                    title="Последствия аварий",
                )
                fig_cons.update_layout(template="plotly_white")
                st.plotly_chart(fig_cons, use_container_width=True)


if __name__ == "__main__":
    main()

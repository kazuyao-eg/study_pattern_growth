import io
import os
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

EFFICIENCY_COL = "学習効率(点数変化/学習日数)"

FINAL_COLS = [
    "ID",
    "学習頻度",
    "学習期間",
    "学習日数",
    "Pre",
    "Post",
    "Post_Pre",
    "Initial_Level",
    "Test_Type",
    EFFICIENCY_COL,
]

FILTER_COLS = [
    "学習頻度",
    "学習期間",
    "学習日数",
    "Pre",
    "Post",
    "Post_Pre",
    "Initial_Level",
    EFFICIENCY_COL,
]

APP_TITLE = "学習パターン（時間）と英語力向上 の分析"


def _round_half_up_2(x) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return float("nan")
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(xf):
        return float("nan")
    return float(Decimal(str(xf)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace(" ", "", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df


def _sheet_test_type(sheet_name: str) -> Optional[str]:
    s = str(sheet_name).lower()
    if "versant" in s:
        return "Versant"
    if "casec" in s:
        return "CASEC"
    return None


def _to_number(s: pd.Series) -> pd.Series:
    if s is None:
        return s
    if pd.api.types.is_numeric_dtype(s):
        return s
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")


def _pearson(x: pd.Series, y: pd.Series) -> float:
    x = _to_number(x)
    y = _to_number(y)
    mask = x.notna() & y.notna()
    if mask.sum() < 2:
        return np.nan
    if x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return np.nan
    return float(x[mask].corr(y[mask], method="pearson"))


@dataclass(frozen=True)
class BuildResult:
    df: Optional[pd.DataFrame]
    error: Optional[str]
    skipped_sheets: List[str]


def build_table_from_excel(uploaded) -> BuildResult:
    if isinstance(uploaded, (bytes, bytearray)):
        uploaded = io.BytesIO(uploaded)
    try:
        uploaded.seek(0)
    except Exception:
        pass
    try:
        xl = pd.ExcelFile(uploaded)
    except Exception:
        return BuildResult(df=None, error="入力データに誤りがあります(列名が正しいか、中身があるかを確認)", skipped_sheets=[])

    tables: List[pd.DataFrame] = []
    skipped: List[str] = []

    for sheet in xl.sheet_names:
        test_type = _sheet_test_type(sheet)
        if test_type is None:
            skipped.append(str(sheet))
            continue

        raw = xl.parse(sheet)
        raw = _clean_columns(raw)
        raw["Test_Type"] = test_type

        if test_type == "Versant":
            rename_map = {
                "Versant_Pre": "Pre",
                "Versant_Post": "Post",
                "Versant_Post_Pre": "Post_Pre",
            }
        else:
            rename_map = {
                "CASEC_Total_Pre": "Pre",
                "CASEC_Total_Post": "Post",
                "CASEC_Total_Post_Pre": "Post_Pre",
            }

        raw = raw.rename(columns=rename_map)

        required_before_eff = [
            "ID",
            "学習頻度",
            "学習期間",
            "学習日数",
            "Pre",
            "Post",
            "Post_Pre",
            "Initial_Level",
            "Test_Type",
        ]
        if any(c not in raw.columns for c in required_before_eff):
            skipped.append(str(sheet))
            continue

        df = raw[required_before_eff].copy()
        df["学習日数"] = _to_number(df["学習日数"])
        df["Post_Pre"] = _to_number(df["Post_Pre"])
        eff = df["Post_Pre"] / df["学習日数"]
        eff = eff.where(df["学習日数"].notna() & (df["学習日数"] != 0), np.nan)
        df[EFFICIENCY_COL] = eff

        df = df[FINAL_COLS].copy()
        tables.append(df)

    if not tables:
        return BuildResult(df=None, error="入力データに誤りがあります(列名が正しいか、中身があるかを確認)", skipped_sheets=skipped)

    out = pd.concat(tables, axis=0, ignore_index=True)
    if list(out.columns) != FINAL_COLS:
        return BuildResult(df=None, error="入力データに誤りがあります(列名が正しいか、中身があるかを確認)", skipped_sheets=skipped)

    if out.isna().all(axis=None):
        return BuildResult(df=None, error="入力データに誤りがあります(列名が正しいか、中身があるかを確認)", skipped_sheets=skipped)

    if out["ID"].isna().all():
        return BuildResult(df=None, error="入力データに誤りがあります(列名が正しいか、中身があるかを確認)", skipped_sheets=skipped)

    return BuildResult(df=out, error=None, skipped_sheets=skipped)


def _is_numeric_series(s: pd.Series) -> bool:
    s2 = _to_number(s)
    return s2.notna().sum() > 0 and s2.notna().mean() >= 0.8


def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    test_types = sorted([x for x in out["Test_Type"].dropna().unique().tolist() if str(x) != ""])
    selected_types = st.sidebar.multiselect("全体のスイッチ（Test_Type）", options=test_types, default=test_types)
    if selected_types:
        out = out[out["Test_Type"].isin(selected_types)]

    for col in FILTER_COLS:
        s = out[col]
        if _is_numeric_series(s):
            s_num = _to_number(s)
            vmin = float(np.nanmin(s_num.values)) if s_num.notna().any() else 0.0
            vmax = float(np.nanmax(s_num.values)) if s_num.notna().any() else 0.0
            if np.isfinite(vmin) and np.isfinite(vmax) and vmin != vmax:
                lo, hi = st.sidebar.slider(
                    f"フィルター: {col}",
                    min_value=float(vmin),
                    max_value=float(vmax),
                    value=(float(vmin), float(vmax)),
                )
                out = out[s_num.between(lo, hi, inclusive="both") | s_num.isna()]
            else:
                st.sidebar.caption(f"フィルター: {col}（絞り込み不可: 値が単一または欠損のみ）")
        else:
            cats = sorted([x for x in s.dropna().astype(str).unique().tolist() if x != ""])
            if len(cats) == 0:
                st.sidebar.caption(f"フィルター: {col}（絞り込み不可: 値なし）")
                continue
            selected = st.sidebar.multiselect(f"フィルター: {col}", options=cats, default=cats)
            if selected:
                out = out[s.astype(str).isin(selected) | s.isna()]

    return out


def _download_csv_button(df: pd.DataFrame, filename: str, label: str) -> None:
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(label=label, data=csv_bytes, file_name=filename, mime="text/csv")


def _bins_6(s: pd.Series) -> Tuple[pd.Series, List[str]]:
    s_num = _to_number(s)
    valid = s_num.dropna()
    if valid.empty:
        binned = pd.Series([np.nan] * len(s_num), index=s_num.index)
        return binned, []

    vmin = float(valid.min())
    vmax = float(valid.max())
    if vmin == vmax:
        label = f"{vmin:g}"
        binned = pd.Series([label if np.isfinite(x) else np.nan for x in s_num], index=s_num.index)
        return binned, [label]

    edges = np.linspace(vmin, vmax, 7)
    labels = []
    for i in range(6):
        a = edges[i]
        b = edges[i + 1]
        if i == 0:
            labels.append(f"{a:g}–{b:g}")
        else:
            labels.append(f"{a:g}–{b:g}")
    binned = pd.cut(s_num, bins=edges, labels=labels, include_lowest=True, duplicates="drop")
    binned = binned.astype(object)
    return binned, labels


def _bins_6_heatmap_rounded(s: pd.Series) -> Tuple[pd.Series, List[str]]:
    """ヒートマップ用: 数値の6分割。境界は小数第3位を四捨五入し第2位まで。"""
    s_num = _to_number(s)
    valid = s_num.dropna()
    if valid.empty:
        binned = pd.Series([np.nan] * len(s_num), index=s_num.index)
        return binned, []

    vmin = _round_half_up_2(valid.min())
    vmax = _round_half_up_2(valid.max())
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        binned = pd.Series([np.nan] * len(s_num), index=s_num.index)
        return binned, []

    if vmin == vmax:
        label = f"{vmin:.2f}"
        binned = pd.Series([label if np.isfinite(float(x)) else np.nan for x in s_num], index=s_num.index)
        return binned, [label]

    edges = [_round_half_up_2(t) for t in np.linspace(vmin, vmax, 7)]
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 0.01
    labels = []
    for i in range(6):
        a, b = edges[i], edges[i + 1]
        labels.append(f"{a:.2f}–{b:.2f}")
    binned = pd.cut(s_num, bins=edges, labels=labels, include_lowest=True, duplicates="drop")
    binned = binned.astype(object)
    return binned, labels


def _tab2_summary(df: pd.DataFrame) -> None:
    st.subheader("サマリー表")
    item = st.selectbox(
        "列を選択",
        options=FILTER_COLS,
        index=FILTER_COLS.index("学習日数") if "学習日数" in FILTER_COLS else 0,
        key="tab2_column_select",
    )

    s = df[item]
    if _is_numeric_series(s):
        group, order = _bins_6(s)
        grouped = df.assign(_group=group)
        columns = [x for x in order if x in grouped["_group"].dropna().unique().tolist()]
    else:
        grouped = df.assign(_group=s.astype(str))
        columns = sorted([x for x in grouped["_group"].dropna().unique().tolist() if x != ""])

    if len(columns) == 0:
        st.info("選択した項目の有効データが不足しています。")
        return

    rows = []
    for g in columns:
        sub = grouped[grouped["_group"] == g]
        pp = _to_number(sub["Post_Pre"])
        id_count = int(sub["ID"].nunique()) if "ID" in sub.columns else int(len(sub))
        rows.append(
            {
                "group": g,
                "n_ids": id_count,
                "median": float(np.nanmedian(pp.values)) if pp.notna().any() else np.nan,
                "mean": float(np.nanmean(pp.values)) if pp.notna().any() else np.nan,
                "var": float(np.nanvar(pp.values, ddof=1)) if pp.notna().sum() >= 2 else np.nan,
                "pearson": _pearson(sub["Post_Pre"], sub["学習日数"]),
            }
        )

    m = pd.DataFrame(rows).set_index("group")
    wide = m[["n_ids", "median", "mean", "var", "pearson"]].T
    wide.index = ["該当人数", "中央値", "平均", "分散", "相関係数（ピアソン）"]

    wide_display = wide.copy()
    for r in ["中央値", "平均", "分散", "相関係数（ピアソン）"]:
        if r in wide_display.index:
            wide_display.loc[r] = wide_display.loc[r].map(
                lambda v: np.nan if (isinstance(v, float) and np.isnan(v)) else _round_half_up_2(v)
            )
    if "該当人数" in wide_display.index:
        wide_display.loc["該当人数"] = wide_display.loc["該当人数"].astype(int)

    st.dataframe(wide_display, use_container_width=True)
    st.caption("※分散・・・データの散らばり具合を表します。数値が大きいほどデータが散らばっています。")
    st.caption("※該当人数が1人以下の場合は分散と相関係数は算出されません。")
    _download_csv_button(
        wide_display.reset_index().rename(columns={"index": "metric"}),
        "tab2_summary.csv",
        "CSVでダウンロード",
    )

    line_df = (
        pd.DataFrame(rows)[["group", "pearson"]]
        .rename(columns={"group": item, "pearson": "相関係数（ピアソン）"})
        .copy()
    )
    line_df["相関係数（ピアソン）"] = line_df["相関係数（ピアソン）"].map(
        lambda v: np.nan if (isinstance(v, float) and np.isnan(v)) else _round_half_up_2(v)
    )
    fig = px.line(line_df, x=item, y="相関係数（ピアソン）", markers=True)
    fig.update_layout(yaxis=dict(tickformat=".2f"))
    st.plotly_chart(fig, use_container_width=True)


def _heatmap_table(df: pd.DataFrame, x: str, y: str, z: str) -> Tuple[pd.DataFrame, str, str]:
    x_s = df[x]
    y_s = df[y]
    z_s = _to_number(df[z]) if _is_numeric_series(df[z]) else pd.Series(df[z].astype(str), index=df.index)

    if _is_numeric_series(x_s):
        xb, x_order = _bins_6_heatmap_rounded(x_s)
        x_label = f"{x}(6分割)"
    else:
        xb = x_s.astype(str)
        x_order = sorted([v for v in xb.dropna().unique().tolist() if v != ""])
        x_label = x

    if _is_numeric_series(y_s):
        yb, y_order = _bins_6_heatmap_rounded(y_s)
        y_label = f"{y}(6分割)"
    else:
        yb = y_s.astype(str)
        y_order = sorted([v for v in yb.dropna().unique().tolist() if v != ""])
        y_label = y

    tmp = df.assign(_x=xb, _y=yb, _z=z_s)
    pivot = tmp.pivot_table(index="_y", columns="_x", values="_z", aggfunc="mean")
    if y_order:
        pivot = pivot.reindex(index=y_order)
    if x_order:
        pivot = pivot.reindex(columns=x_order)
    def _round_cell(v):
        if pd.isna(v):
            return v
        try:
            vf = float(v)
        except (TypeError, ValueError):
            return v
        if not np.isfinite(vf):
            return v
        return _round_half_up_2(vf)

    pivot = pivot.map(_round_cell)
    return pivot, x_label, y_label


def _tab3_heatmap_scatter(df: pd.DataFrame) -> None:
    st.subheader("3項目のヒートマップと散布図")
    c1, c2, c3 = st.columns(3)
    with c1:
        x = st.selectbox(
            "横軸の項目",
            options=FILTER_COLS,
            index=FILTER_COLS.index("学習日数"),
            key="tab3_x_axis",
        )
    with c2:
        y = st.selectbox(
            "縦軸の項目",
            options=FILTER_COLS,
            index=FILTER_COLS.index("Post_Pre"),
            key="tab3_y_axis",
        )
    with c3:
        color = st.selectbox(
            "色(点の大きさ)の項目",
            options=FILTER_COLS,
            index=FILTER_COLS.index(EFFICIENCY_COL),
            key="tab3_color_size",
        )

    pivot, x_label, y_label = _heatmap_table(df, x, y, color)
    if pivot.size == 0 or pivot.dropna(how="all").dropna(axis=1, how="all").empty:
        st.info("ヒートマップを作成できる有効データが不足しています。")
    else:
        fig_hm = px.imshow(
            pivot,
            aspect="equal",
            labels={"x": x_label, "y": y_label, "color": f"平均({color})"},
        )
        fig_hm.update_layout(
            height=800,
            margin=dict(t=130, b=70, l=100, r=50),
            coloraxis_colorbar=dict(
                orientation="h",
                title=dict(text=f"平均({color})", side="top"),
                x=0.5,
                xanchor="center",
                y=1.06,
                yanchor="bottom",
                len=0.55,
                thickness=22,
                tickformat=".2f",
                outlinewidth=0,
            ),
        )
        st.plotly_chart(fig_hm, use_container_width=True, height=800)

    hover_cols = ["ID", "Test_Type"] + FILTER_COLS
    scatter_df = df.copy()
    fig_scatter = px.scatter(
        scatter_df,
        x=_to_number(scatter_df[x]) if _is_numeric_series(scatter_df[x]) else scatter_df[x].astype(str),
        y=_to_number(scatter_df[y]) if _is_numeric_series(scatter_df[y]) else scatter_df[y].astype(str),
        hover_data={c: True for c in hover_cols},
    )
    fig_scatter.update_traces(marker=dict(color="#636EFA", size=8), selector=dict(mode="markers"))
    fig_scatter.update_layout(height=800)
    if _is_numeric_series(scatter_df[x]) and _is_numeric_series(scatter_df[y]):
        fig_scatter.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig_scatter, use_container_width=True, height=800)


def _tab4_scatter(df: pd.DataFrame) -> None:
    st.subheader("2項目の散布図")
    c1, c2 = st.columns(2)
    with c1:
        x = st.selectbox(
            "横軸の項目",
            options=FILTER_COLS,
            index=FILTER_COLS.index("学習日数"),
            key="tab4_x_axis",
        )
    with c2:
        y = st.selectbox(
            "縦軸の項目",
            options=FILTER_COLS,
            index=FILTER_COLS.index(EFFICIENCY_COL),
            key="tab4_y_axis",
        )

    hover_cols = ["ID", "Test_Type"] + FILTER_COLS
    scatter_df = df.copy()
    fig = px.scatter(
        scatter_df,
        x=_to_number(scatter_df[x]) if _is_numeric_series(scatter_df[x]) else scatter_df[x].astype(str),
        y=_to_number(scatter_df[y]) if _is_numeric_series(scatter_df[y]) else scatter_df[y].astype(str),
        hover_data={c: True for c in hover_cols},
    )
    fig.update_layout(height=800)
    if _is_numeric_series(scatter_df[x]) and _is_numeric_series(scatter_df[y]):
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig, use_container_width=True, height=800)


def _check_password() -> bool:
    pw_from_secrets = None
    try:
        pw_from_secrets = st.secrets.get("APP_PASSWORD", None)
    except Exception:
        pw_from_secrets = None

    pw_env = os.environ.get("APP_PASSWORD")
    pw = pw_from_secrets or pw_env

    if not pw:
        st.warning("`APP_PASSWORD` が未設定です。StreamlitのSecrets（または環境変数）に設定してください。")
        return False

    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False

    if st.session_state.auth_ok:
        return True

    entered = st.text_input("パスワード", type="password")
    if entered and entered == pw:
        st.session_state.auth_ok = True
        st.rerun()
    elif entered:
        st.error("パスワードが違います。")
    return False


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    if not _check_password():
        st.stop()

    tabs = st.tabs(
        [
            "Excelファイルのアップロード",
            "サマリー表",
            "3項目のヒートマップと散布図",
            "2項目の散布図",
        ]
    )

    with tabs[0]:
        st.subheader("下記の形式のエクセルファイルをアップロードしてください。")
        st.markdown("")
        st.markdown("　・タブ名は「Versant」(大文字小文字関係なし) or 「Casec」(大文字小文字関係なし)の文字列を含むもの")
        st.markdown("　・列は次の８列の構成：『ID』、『学習頻度』、『学習期間』、『学習日数』、『〇〇_Pre』、『〇〇_Post』、『〇〇_Post-Pre』、『Initial_Level』")
        st.markdown("　・ID列は必ず情報が必要")
        st.markdown("　・ID列以外の他の列は空欄でも可")
        st.markdown("")
        st.markdown("")

        uploaded = st.file_uploader("Excelファイル（.xlsx）", type=["xlsx"], key="excel_uploader_main")
        if uploaded is None:
            st.stop()

        result = build_table_from_excel(uploaded)
        if result.error:
            st.error(result.error)
            if result.skipped_sheets:
                st.caption(f"読み飛ばしたタブ: {', '.join(result.skipped_sheets)}")
            st.stop()

        df = result.df
        st.success(f"読み込み完了: {len(df):,} 行")
        if result.skipped_sheets:
            st.caption(f"読み飛ばしたタブ: {', '.join(result.skipped_sheets)}")
        st.dataframe(df, use_container_width=True)
        _download_csv_button(df, "normalized_table.csv", "整形後テーブルをCSVでダウンロード")
        st.session_state["normalized_df"] = df

    df_all = st.session_state.get("normalized_df")
    if df_all is None:
        st.stop()

    st.sidebar.header("フィルター")
    df_filtered = _apply_filters(df_all)

    with tabs[1]:
        if df_filtered.empty:
            st.warning("フィルター条件に一致するデータがありません。")
        else:
            _tab2_summary(df_filtered)

    with tabs[2]:
        if df_filtered.empty:
            st.warning("フィルター条件に一致するデータがありません。")
        else:
            _tab3_heatmap_scatter(df_filtered)

    with tabs[3]:
        if df_filtered.empty:
            st.warning("フィルター条件に一致するデータがありません。")
        else:
            _tab4_scatter(df_filtered)


if __name__ == "__main__":
    main()


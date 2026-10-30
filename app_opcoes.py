# app_opcoes.py
# pip install streamlit pandas plotly requests
# Rodar: streamlit run app_opcoes.py
import io, os, zipfile, requests, urllib3
from datetime import date, timedelta
import pandas as pd
import plotly.express as px
import streamlit as st

urllib3.disable_warnings()
st.set_page_config(page_title="Opções B3", page_icon="📈", layout="wide")
HIST = "historico_opcoes.csv"
CORES = {"CALL": "#16a34a", "PUT": "#dc2626"}

# ---------- Coleta de dados ----------
def parse(txt):
    reg = []
    for l in txt.splitlines():
        if l[:2] != "01" or l[24:27] not in ("070", "080"):
            continue
        reg.append({"data": l[2:10], "codigo": l[12:24].strip(),
                    "tipo": "CALL" if l[24:27] == "070" else "PUT",
                    "fechamento": int(l[108:121]) / 100, "negocios": int(l[147:152]),
                    "contratos": int(l[152:170]), "volume": int(l[170:188]) / 100,
                    "strike": int(l[188:201]) / 100, "vencimento": l[202:210]})
    return pd.DataFrame(reg)

def baixar(nome):
    url = f"https://bvmf.bmfbovespa.com.br/InstDados/SerHist/{nome}.ZIP"
    r = requests.get(url, verify=False, timeout=180)
    if r.status_code != 200 or not r.content.startswith(b"PK"):
        return None
    z = zipfile.ZipFile(io.BytesIO(r.content))
    return parse(z.read(z.namelist()[0]).decode("latin-1"))

def atualizar(carga_anual=False):
    hist = pd.read_csv(HIST, dtype=str) if os.path.exists(HIST) else pd.DataFrame()
    novos = []
    if carga_anual or hist.empty:
        d = baixar(f"COTAHIST_A{date.today().year}")
        if d is not None:
            novos.append(d.astype(str))
    salvas = set(hist["data"]) if not hist.empty else set()
    for i in range(10, -1, -1):
        d = date.today() - timedelta(days=i)
        if d.weekday() < 5 and f"{d:%Y%m%d}" not in salvas:
            x = baixar(f"COTAHIST_D{d:%d%m%Y}")
            if x is not None and not x.empty:
                novos.append(x.astype(str))
    if novos:
        hist = pd.concat([hist] + novos).drop_duplicates(["data", "codigo"], keep="last")
        hist.to_csv(HIST, index=False)

@st.cache_data
def carregar(_mtime):
    df = pd.read_csv(HIST, dtype=str)
    for c in ["fechamento", "negocios", "contratos", "volume", "strike"]:
        df[c] = pd.to_numeric(df[c])
    df["data"] = pd.to_datetime(df["data"])
    df["vencimento"] = pd.to_datetime(df["vencimento"])
    df["ativo"] = df["codigo"].str[:4]
    df["semana"] = df["data"].dt.to_period("W").dt.start_time
    df["mes"] = df["data"].dt.to_period("M").dt.start_time
    return df

def ranking(d, top):
    g = d.groupby(["codigo", "tipo", "strike", "vencimento"], as_index=False)[
        ["volume", "contratos", "negocios"]].sum()
    return g.sort_values("volume", ascending=False).groupby("tipo").head(top)

def brl(v):
    return f"R$ {v:,.0f}".replace(",", ".")

# ---------- Barra lateral ----------
st.sidebar.title("📈 Opções B3")
if st.sidebar.button("🔄 Atualizar dados da B3", use_container_width=True):
    with st.spinner("Baixando arquivos da B3..."):
        atualizar()
    st.cache_data.clear()
if st.sidebar.button("📦 Carga completa do ano", use_container_width=True):
    with st.spinner("Baixando arquivo anual (pode demorar)..."):
        atualizar(carga_anual=True)
    st.cache_data.clear()

if not os.path.exists(HIST):
    st.info("Nenhum dado ainda. Clique em **Carga completa do ano** na barra lateral.")
    st.stop()

df = carregar(os.path.getmtime(HIST))
ultimo = df["data"].max()
ativos_rank = df.groupby("ativo")["volume"].sum().sort_values(ascending=False)
ativo = st.sidebar.selectbox("Ativo-objeto", ativos_rank.index.tolist())
top = st.sidebar.slider("Top N opções", 5, 30, 10)
ini, fim = st.sidebar.date_input("Período", (df.data.min().date(), ultimo.date()))
st.sidebar.markdown("---\n**Alertas**")
multiplo = st.sidebar.number_input("Múltiplo da média", 1.5, 20.0, 3.0, 0.5)
vol_min = st.sidebar.number_input("Volume mínimo (R$)", 0, 50_000_000, 500_000, 100_000)
janela = st.sidebar.slider("Janela da média (pregões)", 5, 60, 20)

per = df[(df.data >= pd.Timestamp(ini)) & (df.data <= pd.Timestamp(fim))]
a = per[per.ativo == ativo]

# ---------- Cabeçalho e KPIs ----------
st.title(f"Opções de {ativo}")
st.caption(f"Dados até {ultimo:%d/%m/%Y} • Fonte: COTAHIST B3")
vc = a[a.tipo == "CALL"].volume.sum()
vp = a[a.tipo == "PUT"].volume.sum()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Volume total", brl(vc + vp))
k2.metric("Volume CALL", brl(vc))
k3.metric("Volume PUT", brl(vp))
k4.metric("Put/Call ratio", f"{vp / vc:.2f}" if vc else "–")

aba1, aba2, aba3, aba4, aba5 = st.tabs(
    ["📊 Visão geral", "📅 Por vencimento", "🗓️ Semanal", "📆 Mensal", "🚨 Alertas"])

with aba1:
    c1, c2 = st.columns(2)
    t = per.groupby(["ativo", "tipo"], as_index=False)["volume"].sum()
    t = t[t.ativo.isin(ativos_rank.head(15).index)]
    c1.plotly_chart(px.bar(t, x="ativo", y="volume", color="tipo", color_discrete_map=CORES,
                           title="Ações com mais volume em opções"), use_container_width=True)
    d = a.groupby(["data", "tipo"], as_index=False)["volume"].sum()
    c2.plotly_chart(px.line(d, x="data", y="volume", color="tipo", color_discrete_map=CORES,
                            title=f"Volume diário {ativo}"), use_container_width=True)
    st.subheader(f"Mais negociadas no último pregão ({ultimo:%d/%m/%Y})")
    st.dataframe(ranking(df[df.data == ultimo], top), use_container_width=True, hide_index=True)

with aba2:
    vencs = sorted(a.vencimento.unique())
    if vencs:
        futuros = [v for v in vencs if v >= ultimo]
        padrao = vencs.index(futuros[0]) if futuros else len(vencs) - 1
        v = st.selectbox("Vencimento", vencs, index=padrao,
                         format_func=lambda x: pd.Timestamp(x).strftime("%d/%m/%Y"))
        r = ranking(a[a.vencimento == v], top)
        st.plotly_chart(px.bar(r.sort_values("volume"), x="volume", y="codigo", color="tipo",
                               orientation="h", color_discrete_map=CORES, hover_data=["strike"],
                               title=f"Top opções – vencimento {pd.Timestamp(v):%d/%m/%Y}",
                               height=600), use_container_width=True)
        s = a[a.vencimento == v].groupby(["strike", "tipo"], as_index=False)["volume"].sum()
        st.plotly_chart(px.bar(s, x="strike", y="volume", color="tipo", barmode="group",
                               color_discrete_map=CORES, title="Volume por strike"),
                        use_container_width=True)
        st.dataframe(r, use_container_width=True, hide_index=True)

for aba, col, nome in [(aba3, "semana", "semanal"), (aba4, "mes", "mensal")]:
    with aba:
        g = a.groupby([col, "tipo"], as_index=False)["volume"].sum()
        st.plotly_chart(px.bar(g, x=col, y="volume", color="tipo", barmode="group",
                               color_discrete_map=CORES, title=f"Volume {nome} CALL x PUT"),
                        use_container_width=True)
        periodos = sorted(a[col].unique(), reverse=True)
        if periodos:
            p = st.selectbox(f"Período {nome}", periodos, key=col,
                             format_func=lambda x: pd.Timestamp(x).strftime(
                                 "%d/%m/%Y" if col == "semana" else "%m/%Y"))
            c1, c2 = st.columns(2)
            r = ranking(a[a[col] == p], top)
            c1.markdown("🟢 **CALLs mais negociadas**")
            c1.dataframe(r[r.tipo == "CALL"], use_container_width=True, hide_index=True)
            c2.markdown("🔴 **PUTs mais negociadas**")
            c2.dataframe(r[r.tipo == "PUT"], use_container_width=True, hide_index=True)

with aba5:
    pregoes = sorted(df.data.unique())
    ant = pregoes[-(janela + 1):-1]
    hoje = df[df.data == ultimo].groupby(["codigo", "ativo", "tipo", "strike", "vencimento"],
                                         as_index=False)["volume"].sum()
    media = (df[df.data.isin(ant)].groupby("codigo")["volume"].sum()
             / max(len(ant), 1)).rename("media")
    hoje = hoje.merge(media, on="codigo", how="left").fillna({"media": 0})
    hoje["multiplo"] = hoje.volume / hoje.media.replace(0, float("nan"))
    al = hoje[(hoje.volume >= vol_min) & ((hoje.multiplo >= multiplo) | (hoje.media == 0))]
    al = al.sort_values("volume", ascending=False)
    so_ativo = st.toggle(f"Mostrar só {ativo}", value=False)
    if so_ativo:
        al = al[al.ativo == ativo]
    st.metric("Opções com volume anormal hoje", len(al))
    if not al.empty:
        st.plotly_chart(px.scatter(al, x="multiplo", y="volume", color="tipo", size="volume",
                                   hover_name="codigo", hover_data=["strike", "vencimento"],
                                   color_discrete_map=CORES, log_y=True,
                                   title="Volume x quantas vezes acima da média"),
                        use_container_width=True)
        st.dataframe(al, use_container_width=True, hide_index=True)
    else:
        st.success("Nenhum volume anormal com os critérios atuais.")

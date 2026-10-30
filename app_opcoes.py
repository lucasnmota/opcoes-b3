# app_opcoes.py
# Rodar localmente: streamlit run app_opcoes.py
import io, os, gc, json, math, zipfile, requests, urllib3
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

urllib3.disable_warnings()
st.set_page_config(page_title="Opções B3", page_icon="📈", layout="wide")
HIST = "historico_opcoes.parquet"
EST = "estrategias.json"
CORES = {"CALL": "#16a34a", "PUT": "#dc2626"}
COLS = ["data", "codigo", "tipo", "fechamento", "negocios",
        "contratos", "volume", "strike", "vencimento"]
COLS_P = ["Operação", "Tipo", "Código", "Strike", "Prêmio", "Qtd"]

# ---------- Coleta de dados (leitura em partes, pouca memória) ----------
def otimizar(df):
    df["data"] = pd.to_datetime(df["data"], format="%Y%m%d")
    df["vencimento"] = pd.to_datetime(df["vencimento"], format="%Y%m%d", errors="coerce")
    for c in ["fechamento", "volume", "strike"]:
        df[c] = df[c].astype("float64")
    for c in ["negocios", "contratos"]:
        df[c] = df[c].astype("int64")
    return df

def ler_zip(conteudo, desde=None):
    partes, reg = [], []
    z = zipfile.ZipFile(io.BytesIO(conteudo))
    with z.open(z.namelist()[0]) as f:
        for l in f:
            if l[:2] != b"01":
                continue
            mkt = l[24:27]
            if mkt in (b"070", b"080"):
                tipo = "CALL" if mkt == b"070" else "PUT"
                venc = l[202:210].decode()
            elif mkt == b"010" and l[10:12] in (b"02", b"14"):
                tipo, venc = "ACAO", ""   # ação/ETF no lote padrão (preço à vista)
            else:
                continue
            d = l[2:10].decode()
            if desde and d < desde:
                continue
            reg.append((d, l[12:24].decode("latin-1").strip(), tipo,
                        int(l[108:121]) / 100, int(l[147:152]), int(l[152:170]),
                        int(l[170:188]) / 100, int(l[188:201]) / 100, venc))
            if len(reg) >= 200_000:
                partes.append(otimizar(pd.DataFrame(reg, columns=COLS)))
                reg = []
    if reg:
        partes.append(otimizar(pd.DataFrame(reg, columns=COLS)))
    return pd.concat(partes, ignore_index=True) if partes else None

def baixar(nome, desde=None):
    url = f"https://bvmf.bmfbovespa.com.br/InstDados/SerHist/{nome}.ZIP"
    try:
        r = requests.get(url, verify=False, timeout=300)
    except requests.RequestException:
        return None
    if r.status_code != 200 or not r.content.startswith(b"PK"):
        return None
    df = ler_zip(r.content, desde)
    del r
    gc.collect()
    return df

def atualizar(meses, carga_anual=False):
    hist = pd.read_parquet(HIST) if os.path.exists(HIST) else None
    desde = (date.today() - timedelta(days=31 * meses)).strftime("%Y%m%d")
    novos = []
    if carga_anual or hist is None:
        for ano in sorted({int(desde[:4]), date.today().year}):
            d = baixar(f"COTAHIST_A{ano}", desde)
            if d is not None:
                novos.append(d)
    salvas = set(hist["data"].dt.strftime("%Y%m%d")) if hist is not None else set()
    for i in range(10, -1, -1):
        d = date.today() - timedelta(days=i)
        if d.weekday() < 5 and f"{d:%Y%m%d}" not in salvas:
            x = baixar(f"COTAHIST_D{d:%d%m%Y}")
            if x is not None and not x.empty:
                novos.append(x)
    if not novos:
        return False
    tudo = pd.concat(([hist] if hist is not None else []) + novos, ignore_index=True)
    tudo = tudo.drop_duplicates(["data", "codigo"], keep="last")
    tudo = tudo[tudo["data"] >= pd.Timestamp(desde)]
    tudo.to_parquet(HIST, index=False)
    del tudo, novos
    gc.collect()
    return True

@st.cache_data
def carregar(_mtime):
    df = pd.read_parquet(HIST)
    acoes = df[df.tipo == "ACAO"][["data", "codigo", "fechamento", "volume"]].copy()
    df = df[df.tipo != "ACAO"].copy()
    df["ativo"] = df["codigo"].str[:4]
    df["semana"] = df["data"].dt.to_period("W").dt.start_time
    df["mes"] = df["data"].dt.to_period("M").dt.start_time
    ult = pd.concat([df, acoes]).sort_values("data").groupby("codigo")["fechamento"].last()
    return df, acoes, ult

def ranking(d, top):
    g = d.groupby(["codigo", "tipo", "strike", "vencimento"], as_index=False)[
        ["volume", "contratos", "negocios"]].sum()
    return g.sort_values("volume", ascending=False).groupby("tipo").head(top)

def brl(v, casas=0):
    s = f"{v:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"

# ---------- Matemática das estratégias ----------
_erf = np.vectorize(math.erf)

def ncdf(x):
    return 0.5 * (1 + _erf(np.asarray(x, float) / math.sqrt(2)))

def bs(S, K, T, r, vol, tipo):
    """Preço teórico Black-Scholes (aproximação europeia)."""
    S = np.asarray(S, float)
    if T <= 0 or vol <= 0 or K <= 0:
        return np.maximum(S - K, 0) if tipo == "CALL" else np.maximum(K - S, 0)
    raiz = vol * math.sqrt(T)
    d1 = (np.log(S / K) + (r + vol ** 2 / 2) * T) / raiz
    d2 = d1 - raiz
    if tipo == "CALL":
        return S * ncdf(d1) - K * math.exp(-r * T) * ncdf(d2)
    return K * math.exp(-r * T) * ncdf(-d2) - S * ncdf(-d1)

def limpar(p):
    p = pd.DataFrame(p).reindex(columns=COLS_P).copy()
    p = p.dropna(subset=["Operação", "Tipo"])
    for c in ["Strike", "Prêmio", "Qtd"]:
        p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0.0).astype(float)
    p["Código"] = p["Código"].fillna("").astype(str).str.strip().str.upper()
    return p.reset_index(drop=True)

def sinal(p):
    return np.where(p["Operação"] == "Compra", 1.0, -1.0)

def resultado(S, pernas, T=0.0, r=0.0, vol=0.0):
    """Resultado em R$ da estratégia para cada preço S do ativo."""
    S = np.asarray(S, float)
    tot = np.zeros_like(S)
    for sg, (_, p) in zip(sinal(pernas), pernas.iterrows()):
        if p["Tipo"] == "AÇÃO":
            v = S
        elif T > 0:
            v = bs(S, p["Strike"], T, r, vol, p["Tipo"])
        else:
            v = np.maximum(S - p["Strike"], 0) if p["Tipo"] == "CALL" else np.maximum(p["Strike"] - S, 0)
        tot = tot + sg * p["Qtd"] * (v - p["Prêmio"])
    return tot

def breakevens(x, y):
    s = np.sign(y)
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    return [x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i]) for i in idx]

def resumo(pernas, S0):
    custo = float((sinal(pernas) * pernas["Qtd"] * pernas["Prêmio"]).sum())
    larga = np.unique(np.concatenate([np.linspace(0.01, S0 * 3, 3000),
                                      pernas["Strike"][pernas["Strike"] > 0].to_numpy()]))
    v = resultado(larga, pernas)
    inc = v[-1] - v[-2]
    lmax = "Ilimitado" if inc > 1e-9 else brl(v.max(), 2)
    pmax = "Ilimitado" if inc < -1e-9 else brl(v.min(), 2)
    return custo, lmax, pmax, breakevens(larga, v)

# Cada perna: (operação, tipo, distância do strike em "passos" a partir do ATM, multiplicador)
C, V = "Compra", "Venda"
ESTRATEGIAS = {
    # ----- Alta -----
    "Compra de call": ("📈 Alta", "Aposta na alta com risco limitado ao prêmio pago. Ganho ilimitado.",
        [(C, "CALL", 0, 1)]),
    "Venda de put": ("📈 Alta", "Recebe o prêmio se o ativo não cair abaixo do strike. Se cair, você "
        "pode ser exercido e comprar a ação. Usada pra entrar na ação mais barato.",
        [(V, "PUT", -1, 1)]),
    "Trava de alta com call": ("📈 Alta", "Compra uma call e vende outra mais alta. Débito, com ganho e "
        "perda limitados. A trava mais usada pra alta moderada.",
        [(C, "CALL", 0, 1), (V, "CALL", 1, 1)]),
    "Trava de alta com put": ("📈 Alta", "Vende uma put e compra outra mais baixa. Entra com crédito e "
        "ganha se o ativo ficar acima do strike vendido.",
        [(V, "PUT", 0, 1), (C, "PUT", -1, 1)]),
    "Risk reversal": ("📈 Alta", "Vende put fora do dinheiro pra financiar a compra de call. Custo "
        "baixo ou zero, mas com risco forte na queda.",
        [(V, "PUT", -1, 1), (C, "CALL", 1, 1)]),
    "Sintético comprado": ("📈 Alta", "Compra call e vende put no mesmo strike. Imita a compra da ação "
        "usando pouco capital.",
        [(C, "CALL", 0, 1), (V, "PUT", 0, 1)]),
    # ----- Baixa -----
    "Compra de put": ("📉 Baixa", "Aposta na queda ou protege a carteira. Risco limitado ao prêmio.",
        [(C, "PUT", 0, 1)]),
    "Venda de call descoberta": ("📉 Baixa", "Recebe o prêmio se o ativo não subir acima do strike. "
        "Atenção: prejuízo ilimitado na alta.",
        [(V, "CALL", 1, 1)]),
    "Trava de baixa com put": ("📉 Baixa", "Compra uma put e vende outra mais baixa. Débito, com ganho e "
        "perda limitados.",
        [(C, "PUT", 0, 1), (V, "PUT", -1, 1)]),
    "Trava de baixa com call": ("📉 Baixa", "Vende uma call e compra outra mais alta. Entra com crédito e "
        "ganha se o ativo ficar abaixo do strike vendido.",
        [(V, "CALL", 0, 1), (C, "CALL", 1, 1)]),
    "Sintético vendido": ("📉 Baixa", "Vende call e compra put no mesmo strike. Imita a venda da ação.",
        [(V, "CALL", 0, 1), (C, "PUT", 0, 1)]),
    # ----- Neutras -----
    "Iron Condor": ("➖ Mercado parado", "Vende um strangle e compra as pontas pra limitar o risco. Ganha o "
        "crédito se o ativo ficar entre os strikes vendidos. Muito usada pra renda.",
        [(C, "PUT", -2, 1), (V, "PUT", -1, 1), (V, "CALL", 1, 1), (C, "CALL", 2, 1)]),
    "Iron Butterfly": ("➖ Mercado parado", "Vende straddle no dinheiro e compra as pontas. Crédito maior "
        "que o Iron Condor, mas a faixa de lucro é mais estreita.",
        [(C, "PUT", -1, 1), (V, "PUT", 0, 1), (V, "CALL", 0, 1), (C, "CALL", 1, 1)]),
    "Jade Lizard": ("➖ Mercado parado", "Vende put e uma trava de baixa com call. Se o crédito total for "
        "maior que a largura da trava, não há risco na alta, só na queda.",
        [(V, "PUT", -1, 1), (V, "CALL", 1, 1), (C, "CALL", 2, 1)]),
    "Big Lizard": ("➖ Mercado parado", "Vende straddle no dinheiro e compra uma call de proteção. Crédito "
        "alto, risco limitado na alta e aberto na queda.",
        [(V, "PUT", 0, 1), (V, "CALL", 0, 1), (C, "CALL", 2, 1)]),
    "Straddle vendido": ("➖ Mercado parado", "Vende call e put no mesmo strike. Crédito máximo, mas risco "
        "ilimitado se o ativo se mover forte.",
        [(V, "CALL", 0, 1), (V, "PUT", 0, 1)]),
    "Strangle vendido": ("➖ Mercado parado", "Vende call e put fora do dinheiro. Faixa de lucro larga, "
        "com risco ilimitado nos extremos.",
        [(V, "CALL", 1, 1), (V, "PUT", -1, 1)]),
    "Borboleta com call": ("➖ Mercado parado", "Compra 1, vende 2 e compra 1 call. Custo baixo, lucro "
        "máximo se o ativo vencer no strike do meio.",
        [(C, "CALL", -1, 1), (V, "CALL", 0, 2), (C, "CALL", 1, 1)]),
    "Borboleta com put": ("➖ Mercado parado", "Mesma ideia da borboleta, montada com puts.",
        [(C, "PUT", -1, 1), (V, "PUT", 0, 2), (C, "PUT", 1, 1)]),
    "Condor com call": ("➖ Mercado parado", "Como a borboleta, mas com o miolo mais largo. Débito "
        "pequeno e faixa de lucro maior.",
        [(C, "CALL", -2, 1), (V, "CALL", -1, 1), (V, "CALL", 1, 1), (C, "CALL", 2, 1)]),
    "Ratio spread com call (1x2)": ("➖ Mercado parado", "Compra 1 call e vende 2 mais altas. Ganha com "
        "alta leve; risco ilimitado em alta forte.",
        [(C, "CALL", 0, 1), (V, "CALL", 1, 2)]),
    "Ratio spread com put (1x2)": ("➖ Mercado parado", "Compra 1 put e vende 2 mais baixas. Ganha com "
        "queda leve; risco grande em queda forte.",
        [(C, "PUT", 0, 1), (V, "PUT", -1, 2)]),
    # ----- Volatilidade -----
    "Straddle comprado": ("💥 Movimento forte", "Compra call e put no mesmo strike. Ganha se o ativo se "
        "mover forte para qualquer lado, como em balanços.",
        [(C, "CALL", 0, 1), (C, "PUT", 0, 1)]),
    "Strangle comprado": ("💥 Movimento forte", "Compra call e put fora do dinheiro. Mais barato que o "
        "straddle, mas precisa de um movimento maior.",
        [(C, "CALL", 1, 1), (C, "PUT", -1, 1)]),
    "Backspread com call": ("💥 Movimento forte", "Vende 1 call e compra 2 mais altas. Ganha muito em "
        "alta forte, perde se o ativo subir pouco.",
        [(V, "CALL", 0, 1), (C, "CALL", 1, 2)]),
    "Backspread com put": ("💥 Movimento forte", "Vende 1 put e compra 2 mais baixas. Ganha muito em "
        "queda forte.",
        [(V, "PUT", 0, 1), (C, "PUT", -1, 2)]),
    # ----- Com a ação -----
    "Venda coberta": ("🏦 Com a ação", "Tem a ação e vende call acima do preço. Gera renda, mas limita "
        "o ganho na alta.",
        [(C, "AÇÃO", 0, 1), (V, "CALL", 1, 1)]),
    "Financiamento (call ITM)": ("🏦 Com a ação", "Compra a ação e vende call dentro do dinheiro. Muito "
        "usada na B3: taxa de retorno conhecida se for exercido.",
        [(C, "AÇÃO", 0, 1), (V, "CALL", -1, 1)]),
    "Put protetora": ("🏦 Com a ação", "Tem a ação e compra put como seguro contra queda.",
        [(C, "AÇÃO", 0, 1), (C, "PUT", -1, 1)]),
    "Collar": ("🏦 Com a ação", "Ação com put de proteção, financiada pela venda de call. Trava "
        "o resultado numa faixa.",
        [(C, "AÇÃO", 0, 1), (C, "PUT", -1, 1), (V, "CALL", 1, 1)]),
}
CATEGORIAS = list(dict.fromkeys(v[0] for v in ESTRATEGIAS.values()))
PRESETS = list(ESTRATEGIAS)

def montar_preset(nome, S, passo, T, r, vol, qtd, cod_acao=""):
    atm = round(S * 2) / 2
    linhas = []
    for op, tipo, off, mult in ESTRATEGIAS[nome][2]:
        if tipo == "AÇÃO":
            linhas.append({"Operação": op, "Tipo": "AÇÃO", "Código": cod_acao, "Strike": 0.0,
                           "Prêmio": round(S, 2), "Qtd": float(qtd * mult)})
            continue
        K = round(max(atm + off * passo, 0.01), 2)
        linhas.append({"Operação": op, "Tipo": tipo, "Código": "", "Strike": K,
                       "Prêmio": round(float(bs(S, K, max(T, 1 / 252), r, vol, tipo)), 2),
                       "Qtd": float(qtd * mult)})
    return limpar(pd.DataFrame(linhas))

def grafico_payoff(pernas, S0, T=0.0, r=0.0, vol=0.0, titulo="Payoff da estratégia",
                   ponto=None):
    grade = np.linspace(S0 * 0.6, S0 * 1.4, 400)
    venc = resultado(grade, pernas)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grade, y=np.where(venc >= 0, venc, 0), fill="tozeroy",
                             line=dict(width=0), fillcolor="rgba(22,163,74,0.15)",
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=grade, y=np.where(venc < 0, venc, 0), fill="tozeroy",
                             line=dict(width=0), fillcolor="rgba(220,38,38,0.15)",
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=grade, y=venc, name="No vencimento",
                             line=dict(width=3, color="#2563eb")))
    if T > 0:
        fig.add_trace(go.Scatter(x=grade, y=resultado(grade, pernas, T, r, vol),
                                 name="Hoje (teórico)", line=dict(dash="dash", color="#f59e0b")))
    if ponto is not None:
        fig.add_trace(go.Scatter(x=[ponto[0]], y=[ponto[1]], mode="markers",
                                 marker=dict(size=13, color="#111827"), name="Posição atual"))
    fig.add_hline(y=0, line_color="gray", line_width=1)
    fig.add_vline(x=S0, line_dash="dot", line_color="gray", annotation_text="Preço atual")
    fig.update_layout(title=titulo, xaxis_title="Preço do ativo (R$)",
                      yaxis_title="Resultado (R$)", hovermode="x unified", height=480)
    return fig

# ---------- Estratégias salvas ----------
def salvar_est():
    try:
        with open(EST, "w", encoding="utf-8") as f:
            json.dump(st.session_state.estrategias, f, ensure_ascii=False, default=float, indent=1)
    except OSError:
        pass

if "estrategias" not in st.session_state:
    try:
        with open(EST, encoding="utf-8") as f:
            st.session_state.estrategias = json.load(f)
    except Exception:
        st.session_state.estrategias = []
if "pernas" not in st.session_state:
    st.session_state.pernas = limpar(pd.DataFrame(columns=COLS_P))
    st.session_state.ed_v = 0

# ---------- Barra lateral ----------
st.sidebar.title("📈 Opções B3")
meses = st.sidebar.slider("Meses de histórico", 1, 6, 3,
                          help="Mais meses = mais dados, mas a carga fica mais lenta.")
if st.sidebar.button("🔄 Atualizar dados da B3", use_container_width=True):
    with st.spinner("Baixando arquivos da B3..."):
        ok = atualizar(meses)
    st.cache_data.clear()
    if not ok:
        st.sidebar.warning("Nenhum arquivo novo encontrado na B3.")
if st.sidebar.button("📦 Carga completa", use_container_width=True):
    with st.spinner("Baixando histórico da B3 (pode levar alguns minutos)..."):
        ok = atualizar(meses, carga_anual=True)
    st.cache_data.clear()
    if not ok:
        st.sidebar.error("Não consegui baixar os arquivos da B3. Tente de novo em alguns minutos.")

if not os.path.exists(HIST):
    st.info("Nenhum dado ainda. Clique em **Carga completa** na barra lateral.")
    st.stop()

df, acoes, precos = carregar(os.path.getmtime(HIST))
ultimo = df["data"].max()

def spot_de(raiz):
    if acoes.empty:
        return None, ""
    x = acoes[(acoes.data == acoes.data.max()) & acoes.codigo.str.startswith(raiz)]
    if x.empty:
        return None, ""
    r = x.sort_values("volume").iloc[-1]
    return float(r.fechamento), r.codigo

ativos_rank = df.groupby("ativo")["volume"].sum().sort_values(ascending=False)
ativo = st.sidebar.selectbox("Ativo-objeto", ativos_rank.index.tolist())
top = st.sidebar.slider("Top N opções", 5, 30, 10)
periodo = st.sidebar.date_input("Período", (df.data.min().date(), ultimo.date()))
ini, fim = (periodo if isinstance(periodo, (list, tuple)) and len(periodo) == 2
            else (df.data.min().date(), ultimo.date()))
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

(aba1, aba_pc, aba2, aba3, aba4, aba5, aba_sim, aba_acomp) = st.tabs(
    ["📊 Visão geral", "⚖️ Put/Call", "📅 Por vencimento", "🗓️ Semanal", "📆 Mensal",
     "🚨 Alertas", "🧮 Simulador", "📒 Minhas estratégias"])

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

# ---------- Put/Call ratio diário ----------
with aba_pc:
    p1, p2 = st.columns([2, 1])
    base_pc = p1.radio("Calcular por", ["Volume financeiro (R$)", "Contratos"], horizontal=True)
    mm = p2.slider("Média móvel (pregões)", 5, 60, 20)
    campo = "volume" if base_pc.startswith("Volume") else "contratos"
    d = (a.pivot_table(index="data", columns="tipo", values=campo, aggfunc="sum")
           .reindex(columns=["CALL", "PUT"]).fillna(0))
    d = d[d.CALL > 0]
    if d.empty:
        st.info("Sem dados suficientes de calls e puts para esse ativo no período.")
    else:
        d["ratio"] = d.PUT / d.CALL
        d["media"] = d.ratio.rolling(mm, min_periods=3).mean()
        dp = d.ratio.rolling(mm, min_periods=3).std()
        d["sup"] = d.media + 2 * dp
        d["inf"] = (d.media - 2 * dp).clip(lower=0)
        atual_r, med = d.ratio.iloc[-1], d.media.iloc[-1]
        pct = (d.ratio <= atual_r).mean() * 100
        m1, m2, m3 = st.columns(3)
        m1.metric("P/C do último pregão", f"{atual_r:.2f}",
                  f"{atual_r - med:+.2f} vs média" if pd.notna(med) else None,
                  delta_color="inverse")
        m2.metric(f"Média de {mm} pregões", f"{med:.2f}" if pd.notna(med) else "–")
        m3.metric("Percentil no período", f"{pct:.0f}%",
                  help="Em quantos % dos pregões do período o P/C ficou abaixo do valor atual.")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d.index, y=d.sup, line=dict(width=0), showlegend=False,
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=d.index, y=d.inf, line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(37,99,235,0.12)", name="Faixa normal (±2 desvios)"))
        fig.add_trace(go.Scatter(x=d.index, y=d.ratio, name="Put/Call diário",
                                 line=dict(color="#111827", width=2)))
        fig.add_trace(go.Scatter(x=d.index, y=d.media, name=f"Média {mm}",
                                 line=dict(color="#2563eb", dash="dash")))
        fig.update_layout(title=f"Put/Call ratio diário – {ativo}", hovermode="x unified",
                          yaxis_title="Puts ÷ Calls", height=450)
        st.plotly_chart(fig, use_container_width=True)
        if pd.notna(d.sup.iloc[-1]) and atual_r > d.sup.iloc[-1]:
            st.warning("O P/C está **acima da faixa normal**: há mais procura por puts que o "
                       "habitual para este ativo, sinal de proteção ou pessimismo.")
        elif pd.notna(d.inf.iloc[-1]) and atual_r < d.inf.iloc[-1]:
            st.success("O P/C está **abaixo da faixa normal**: calls dominando mais que o "
                       "habitual, sinal de otimismo acima da média.")
        else:
            st.caption("O P/C está dentro da faixa normal do ativo.")
        r_at = (per.pivot_table(index="ativo", columns="tipo", values=campo, aggfunc="sum")
                  .reindex(columns=["CALL", "PUT"]).fillna(0))
        r_at = r_at.loc[r_at.index.isin(ativos_rank.head(20).index)]
        r_at = r_at[r_at.CALL > 0].assign(ratio=lambda x: x.PUT / x.CALL).sort_values("ratio")
        st.plotly_chart(px.bar(r_at.reset_index(), x="ratio", y="ativo", orientation="h",
                               title="Put/Call no período – 20 ativos mais negociados",
                               labels={"ratio": "Puts ÷ Calls", "ativo": ""}, height=550),
                        use_container_width=True)

with aba2:
    vencs = sorted(a.vencimento.dropna().unique())
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

# ---------- Simulador de estratégias ----------
with aba_sim:
    spot0, cod_spot = spot_de(ativo)
    vv_fut = sorted(df[(df.ativo == ativo) & (df.vencimento > ultimo)].vencimento.dropna().unique())
    venc_pad = pd.Timestamp(vv_fut[0]).date() if vv_fut else ultimo.date() + timedelta(days=30)
    dias_pad = int(min(max(np.busday_count(ultimo.date(), venc_pad), 0), 500))

    st.markdown(f"**Ativo:** {cod_spot or ativo}" +
                (f" • último preço {brl(spot0, 2)}" if spot0 else
                 " • preço à vista não encontrado, informe manualmente"))
    c = st.columns(5)
    S0 = c[0].number_input("Preço do ativo (R$)", 0.01, 100000.0, float(spot0 or 30.0), 0.10)
    dias = c[1].number_input("Dias úteis até o vencimento", 0, 500, dias_pad)
    vol = c[2].number_input("Volatilidade (% a.a.)", 1.0, 300.0, 35.0, 1.0) / 100
    taxa = c[3].number_input("Juros (% a.a.)", 0.0, 50.0, 13.0, 0.25) / 100
    qtd_pad = c[4].number_input("Qtd. padrão", 100, 10_000_000, 100, 100)
    T, r_cont = dias / 252, math.log(1 + taxa)

    cp0, cp1, cp2, cp3, cp4 = st.columns([1.6, 2.4, 1.3, 1, 1])
    cat = cp0.selectbox("Cenário", CATEGORIAS)
    preset = cp1.selectbox("Estratégia", [k for k, v in ESTRATEGIAS.items() if v[0] == cat])
    passo = cp2.number_input("Distância entre strikes (R$)", 0.01, 10000.0,
                             float(max(round(S0 * 0.05 * 2) / 2, 0.5)), 0.5)
    cp3.write(""); cp4.write("")
    if cp3.button("Montar", use_container_width=True):
        st.session_state.pernas = montar_preset(preset, S0, passo, T, r_cont, vol, qtd_pad, cod_spot)
        st.session_state.ed_v += 1
        st.rerun()
    if cp4.button("Limpar", use_container_width=True):
        st.session_state.pernas = limpar(pd.DataFrame(columns=COLS_P))
        st.session_state.ed_v += 1
        st.rerun()

    st.info(f"**{preset}:** {ESTRATEGIAS[preset][1]}")
    st.caption("Edite as pernas na tabela: troque strike, prêmio ou quantidade, ou adicione "
               "linhas. Prêmios das estratégias prontas são teóricos (Black-Scholes).")
    editado = st.data_editor(
        st.session_state.pernas, key=f"ed_{st.session_state.ed_v}", num_rows="dynamic",
        use_container_width=True, hide_index=True,
        column_config={
            "Operação": st.column_config.SelectboxColumn(options=["Compra", "Venda"],
                                                         required=True, default="Compra"),
            "Tipo": st.column_config.SelectboxColumn(options=["CALL", "PUT", "AÇÃO"],
                                                     required=True, default="CALL"),
            "Código": st.column_config.TextColumn(help="Código da opção ou da ação (opcional, "
                                                       "usado no acompanhamento)"),
            "Strike": st.column_config.NumberColumn(format="%.2f", min_value=0.0, default=0.0),
            "Prêmio": st.column_config.NumberColumn("Prêmio / preço", format="%.2f",
                                                    min_value=0.0, default=0.0),
            "Qtd": st.column_config.NumberColumn(min_value=0, step=100, default=100),
        })
    pernas = limpar(editado)

    with st.expander("➕ Adicionar uma opção real do último pregão"):
        hoje_a = df[(df.data == ultimo) & (df.ativo == ativo)]
        vv = sorted(hoje_a.vencimento.dropna().unique())
        if not vv:
            st.caption("Esse ativo não teve opções negociadas no último pregão.")
        else:
            e1, e2, e3, e4, e5 = st.columns([1.2, 2.6, 1, 1, 1])
            vsel = e1.selectbox("Vencimento", vv, key="add_v",
                                format_func=lambda x: pd.Timestamp(x).strftime("%d/%m/%Y"))
            ops = (hoje_a[hoje_a.vencimento == vsel]
                   .groupby(["codigo", "tipo", "strike"], as_index=False)
                   .agg(fechamento=("fechamento", "last"), volume=("volume", "sum"))
                   .sort_values(["tipo", "strike"]).reset_index(drop=True))
            i_op = e2.selectbox("Opção", range(len(ops)), key="add_o",
                                format_func=lambda i: f"{ops.codigo[i]} · {ops.tipo[i]} · "
                                                      f"K {ops.strike[i]:.2f} · R$ {ops.fechamento[i]:.2f}")
            op = e3.selectbox("Operação", ["Compra", "Venda"], key="add_op")
            q = e4.number_input("Qtd", 1, 10_000_000, int(qtd_pad), 100, key="add_q")
            e5.write("")
            if e5.button("Adicionar", use_container_width=True):
                o = ops.iloc[i_op]
                nova = pd.DataFrame([{"Operação": op, "Tipo": o.tipo, "Código": o.codigo,
                                      "Strike": o.strike, "Prêmio": o.fechamento, "Qtd": float(q)}])
                st.session_state.pernas = limpar(pd.concat([pernas, nova], ignore_index=True))
                st.session_state.ed_v += 1
                st.rerun()
            st.caption(f"Faltam {np.busday_count(ultimo.date(), pd.Timestamp(vsel).date())} "
                       "dias úteis para esse vencimento.")

    if pernas.empty:
        st.info("Escolha uma estratégia pronta e clique em **Montar**, ou adicione pernas na tabela.")
    else:
        custo, lmax, pmax, bes = resumo(pernas, S0)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Montagem", ("Débito " if custo > 0 else "Crédito ") + brl(abs(custo), 2))
        m2.metric("Lucro máximo", lmax)
        m3.metric("Prejuízo máximo", pmax)
        m4.metric("Break-even", " / ".join(f"{b:.2f}" for b in bes) if bes else "–")
        st.plotly_chart(grafico_payoff(pernas, S0, T, r_cont, vol), use_container_width=True)
        st.caption("Linha azul: resultado no vencimento. Linha tracejada: resultado teórico hoje, "
                   "pela volatilidade e prazo informados. Opções de ações na B3 são americanas; "
                   "o modelo é uma aproximação.")

        s1, s2, s3 = st.columns([2, 1, 1])
        nome_est = s1.text_input("Nome da estratégia", f"{preset} {ativo}")
        venc_est = s2.date_input("Vencimento", venc_pad, key="venc_est")
        s3.write("")
        if s3.button("💾 Salvar para acompanhar", use_container_width=True):
            st.session_state.estrategias.append({
                "id": datetime.now().strftime("%Y%m%d%H%M%S%f"), "nome": nome_est,
                "ativo": ativo, "spot_entrada": S0, "criada": date.today().isoformat(),
                "vencimento": venc_est.isoformat(), "pernas": pernas.to_dict("records")})
            salvar_est()
            st.success("Estratégia salva! Veja na aba **Minhas estratégias**.")

# ---------- Acompanhamento ----------
with aba_acomp:
    ests = st.session_state.estrategias
    b1, b2 = st.columns(2)
    b1.download_button("⬇️ Baixar backup das estratégias",
                       json.dumps(ests, ensure_ascii=False, default=float, indent=1),
                       "estrategias.json", "application/json", use_container_width=True)
    up = b2.file_uploader("Restaurar backup", type="json", label_visibility="collapsed")
    if up is not None and b2.button("Restaurar este backup", use_container_width=True):
        st.session_state.estrategias = json.load(up)
        salvar_est()
        st.rerun()
    st.caption("No servidor gratuito, as estratégias podem se perder quando o app reinicia. "
               "Baixe o backup de vez em quando.")

    if not ests:
        st.info("Nenhuma estratégia salva ainda. Monte uma no **Simulador** e clique em salvar.")
    else:
        linhas, detalhes = [], []
        for e in ests:
            p = limpar(pd.DataFrame(e["pernas"]))
            sg = sinal(p)
            atual = p["Código"].map(precos)
            sem_preco = atual.isna()
            atual = atual.fillna(p["Prêmio"])
            p["Preço atual"] = atual
            p["P&L (R$)"] = sg * p["Qtd"] * (atual - p["Prêmio"])
            custo = float((sg * p["Qtd"] * p["Prêmio"]).sum())
            pl = float(p["P&L (R$)"].sum())
            try:
                dias_r = int(np.busday_count(date.today(), date.fromisoformat(e["vencimento"])))
            except Exception:
                dias_r = None
            linhas.append({"Estratégia": e["nome"], "Ativo": e["ativo"], "Montada em": e["criada"],
                           "Vencimento": e["vencimento"], "Dias úteis": dias_r,
                           "Montagem (R$)": custo, "P&L (R$)": pl,
                           "P&L (%)": pl / abs(custo) * 100 if custo else None})
            detalhes.append((e, p, sem_preco, pl))

        tab = pd.DataFrame(linhas)
        st.metric("Resultado total das estratégias", brl(tab["P&L (R$)"].sum(), 2))
        st.dataframe(tab, use_container_width=True, hide_index=True, column_config={
            "Montagem (R$)": st.column_config.NumberColumn(format="R$ %.2f"),
            "P&L (R$)": st.column_config.NumberColumn(format="R$ %.2f"),
            "P&L (%)": st.column_config.NumberColumn(format="%.1f%%")})
        st.caption(f"Preços atuais pelo último fechamento disponível ({ultimo:%d/%m/%Y}). "
                   "Clique em **Atualizar dados da B3** para trazer o pregão mais recente.")

        for e, p, sem_preco, pl in detalhes:
            icone = "🟢" if pl >= 0 else "🔴"
            with st.expander(f"{icone} {e['nome']} • {brl(pl, 2)}"):
                if sem_preco.any():
                    st.caption("⚠️ Pernas sem código ou sem negócio recente estão avaliadas pelo "
                               "preço de entrada. Preencha o código na hora de montar para "
                               "acompanhar de verdade.")
                st.dataframe(p, use_container_width=True, hide_index=True, column_config={
                    "Prêmio": st.column_config.NumberColumn("Entrada", format="%.2f"),
                    "Preço atual": st.column_config.NumberColumn(format="%.2f"),
                    "P&L (R$)": st.column_config.NumberColumn(format="R$ %.2f")})
                spot_now = spot_de(e["ativo"])[0] or float(e.get("spot_entrada", 0)) or 1.0
                st.plotly_chart(grafico_payoff(p[COLS_P], spot_now, titulo="Payoff no vencimento",
                                               ponto=(spot_now, pl)),
                                use_container_width=True, key=f"g_{e['id']}")
                if st.button("🗑️ Excluir estratégia", key=f"del_{e['id']}"):
                    st.session_state.estrategias = [x for x in ests if x["id"] != e["id"]]
                    salvar_est()
                    st.rerun()

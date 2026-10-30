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
                      yaxis_title="Resultado (R$)", hovermode="x unified",
                      xaxis_tickformat=",.2f", yaxis_tickformat=",.0f")
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

# ---------- Formatação ----------
def num(v, casas=2):
    return f"{v:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")

def brl_c(v):
    """Valor em reais no formato curto: R$ 12,65 bi, R$ 3,4 mi..."""
    a = abs(v)
    for lim, suf in [(1e9, "bi"), (1e6, "mi"), (1e3, "mil")]:
        if a >= lim:
            return f"R$ {num(v / lim, 2)} {suf}"
    return f"R$ {num(v, 2)}"

LAB = {"volume": "Volume (R$)", "tipo": "Tipo", "ativo": "Ativo", "data": "Data",
       "semana": "Semana", "mes": "Mês", "strike": "Strike (R$)", "codigo": "Opção",
       "multiplo": "Vezes acima da média", "contratos": "Contratos", "vencimento": "Vencimento"}

def estilo(fig, altura=430):
    fig.update_layout(separators=",.", height=altura, margin=dict(l=10, r=10, t=60, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
                      title=dict(font=dict(size=16)))
    return fig

def grafico(cont, fig, altura=430, key=None):
    cont.plotly_chart(estilo(fig, altura), width="stretch", key=key)

CFG = {
    "codigo": st.column_config.TextColumn("Opção"),
    "tipo": st.column_config.TextColumn("Tipo"),
    "ativo": st.column_config.TextColumn("Ativo"),
    "strike": st.column_config.NumberColumn("Strike", format="localized"),
    "vencimento": st.column_config.DateColumn("Vencimento", format="DD/MM/YYYY"),
    "volume": st.column_config.NumberColumn("Volume (R$)", format="compact"),
    "contratos": st.column_config.NumberColumn("Contratos", format="localized"),
    "negocios": st.column_config.NumberColumn("Negócios", format="localized"),
    "media": st.column_config.NumberColumn("Média diária (R$)", format="compact"),
    "multiplo": st.column_config.NumberColumn("Vezes a média", format="%.1fx"),
    "fechamento": st.column_config.NumberColumn("Último preço", format="localized"),
}

def tabela(cont, d, extra=None):
    cfg = {k: v for k, v in CFG.items() if k in d.columns}
    cfg.update(extra or {})
    cont.dataframe(d, width="stretch", hide_index=True, column_config=cfg)

# ---------- Cotações ----------
POPULARES = ["PETR4", "PETR3", "VALE3", "BOVA11", "ITUB4", "BBDC4", "BBAS3", "B3SA3",
             "ABEV3", "PRIO3", "WEGE3", "MGLU3", "SUZB3", "GGBR4", "CSNA3", "USIM5",
             "ELET3", "ITSA4", "RENT3", "LREN3", "HAPV3", "CMIG4", "BBSE3", "EMBR3",
             "RADL3", "VBBR3", "JBSS3", "BRFS3", "CSAN3", "COGN3", "AZUL4", "SMAL11"]

@st.cache_data(ttl=60, show_spinner=False)
def cotacao_yahoo(tk):
    try:
        import yfinance as yf
        t = yf.Ticker(f"{tk}.SA")
        dia = t.history(period="10d", interval="1d")
        if dia.empty:
            return None
        intr = t.history(period="1d", interval="1m")
        serie = intr if not intr.empty else dia
        preco, hora = float(serie["Close"].iloc[-1]), serie.index[-1]
        if getattr(hora, "tzinfo", None) is not None:
            hora = hora.tz_convert("America/Sao_Paulo")
        ant = dia[dia.index.date < hora.date()]["Close"]
        prev = float(ant.iloc[-1]) if len(ant) else None
        return {"preco": preco, "var": (preco / prev - 1) * 100 if prev else None,
                "hora": hora, "fonte": "Yahoo Finance, atraso de ~15 min"}
    except Exception:
        return None

def cotacao_b3(tk):
    x = acoes[acoes.codigo == tk].sort_values("data")
    if x.empty:
        return None
    preco = float(x.fechamento.iloc[-1])
    prev = float(x.fechamento.iloc[-2]) if len(x) > 1 else None
    return {"preco": preco, "var": (preco / prev - 1) * 100 if prev else None,
            "hora": x.data.iloc[-1], "fonte": "último fechamento da B3"}

def cotacao(tk):
    if not tk:
        return None
    return cotacao_yahoo(tk) or cotacao_b3(tk)

def texto_hora(q):
    h = pd.Timestamp(q["hora"])
    return h.strftime("%d/%m %H:%M") if (h.hour or h.minute) else h.strftime("%d/%m/%Y")

# ---------- Barra lateral ----------
st.sidebar.title("📈 Opções B3")
meses = st.sidebar.slider("Meses de histórico", 1, 6, 3,
                          help="Mais meses = mais dados, mas a carga fica mais lenta.")
b_up, b_full = st.sidebar.columns(2)
if b_up.button("🔄 Atualizar", width="stretch", help="Baixa os pregões que faltam"):
    with st.spinner("Baixando arquivos da B3..."):
        ok = atualizar(meses)
    st.cache_data.clear()
    if not ok:
        st.sidebar.warning("Nenhum arquivo novo encontrado na B3.")
if b_full.button("📦 Carga completa", width="stretch", help="Baixa todo o histórico de novo"):
    with st.spinner("Baixando histórico da B3 (pode levar alguns minutos)..."):
        ok = atualizar(meses, carga_anual=True)
    st.cache_data.clear()
    if not ok:
        st.sidebar.error("Não consegui baixar os arquivos da B3. Tente de novo em alguns minutos.")

if not os.path.exists(HIST):
    st.info("Nenhum dado ainda. Clique em **📦 Carga completa** na barra lateral.")
    st.stop()

df, acoes, precos = carregar(os.path.getmtime(HIST))
ultimo = df["data"].max()
ativos_rank = df.groupby("ativo")["volume"].sum().sort_values(ascending=False)

def spot_de(raiz):
    if acoes.empty:
        return None, ""
    x = acoes[(acoes.data == acoes.data.max()) & acoes.codigo.str.startswith(raiz)]
    if x.empty:
        return None, ""
    r = x.sort_values("volume").iloc[-1]
    return float(r.fechamento), r.codigo

# Lista de tickers: ações com opções, ordenadas pelo volume em opções
raizes = list(ativos_rank.index)
if not acoes.empty:
    vol_acao = acoes.groupby("codigo")["volume"].sum()
    vol_acao = vol_acao[vol_acao.index.str.match(r"^[A-Z0-9]{4}\d{1,2}$")
                        & vol_acao.index.str[:4].isin(raizes)]
    ordem = {r: i for i, r in enumerate(raizes)}
    tickers = sorted(vol_acao.index, key=lambda t: (ordem[t[:4]], -vol_acao[t]))
else:
    tickers = []
tickers += [t for t in POPULARES if t not in tickers and t[:4] in raizes]
if not tickers:
    tickers = POPULARES[:]
ticker = st.sidebar.selectbox(
    "Ativo", tickers, index=tickers.index("PETR4") if "PETR4" in tickers else 0,
    accept_new_options=True, help="Escolha da lista ou digite o código, como PETR4 ou VALE3.")
ticker = (ticker or "PETR4").strip().upper()
ativo = ticker[:4]
if acoes.empty:
    st.sidebar.warning("Clique em **📦 Carga completa** uma vez para trazer os preços das ações.")

top = st.sidebar.slider("Top N opções", 5, 30, 10)
periodo = st.sidebar.date_input("Período", (df.data.min().date(), ultimo.date()),
                                format="DD/MM/YYYY")
ini, fim = (periodo if isinstance(periodo, (list, tuple)) and len(periodo) == 2
            else (df.data.min().date(), ultimo.date()))
with st.sidebar.expander("🚨 Critérios dos alertas"):
    multiplo = st.number_input("Múltiplo da média", 1.5, 20.0, 3.0, 0.5)
    vol_min = st.number_input("Volume mínimo (R$)", 0, 50_000_000, 500_000, 100_000)
    janela = st.slider("Janela da média (pregões)", 5, 60, 20)

per = df[(df.data >= pd.Timestamp(ini)) & (df.data <= pd.Timestamp(fim))]
a = per[per.ativo == ativo]

# ---------- Cabeçalho e KPIs ----------
q = cotacao(ticker)
h1, h2 = st.columns([4, 1])
h1.title(ticker)
h1.caption(f"Opções com raiz **{ativo}** • histórico de opções até {ultimo:%d/%m/%Y} "
           "• fonte: COTAHIST B3")
h2.write("")
if h2.button("🔄 Atualizar cotação", width="stretch"):
    cotacao_yahoo.clear()
    st.rerun()
if ativo not in raizes:
    st.warning(f"Não encontrei opções de {ticker} no histórico carregado.")

vc = a[a.tipo == "CALL"].volume.sum()
vp = a[a.tipo == "PUT"].volume.sum()
k0, k1, k2, k3, k4 = st.columns(5)
if q:
    k0.metric(f"Cotação {ticker}", f"R$ {num(q['preco'])}",
              f"{num(q['var'])}%" if q["var"] is not None else None,
              help=f"{q['fonte']} • {texto_hora(q)}")
else:
    k0.metric(f"Cotação {ticker}", "–", help="Cotação não encontrada.")
k1.metric("Volume em opções", brl_c(vc + vp), help=brl(vc + vp))
k2.metric("Volume CALL", brl_c(vc), help=brl(vc))
k3.metric("Volume PUT", brl_c(vp), help=brl(vp))
k4.metric("Put/Call ratio", num(vp / vc) if vc else "–")
if q:
    st.caption(f"Cotação: {q['fonte']} • {texto_hora(q)}")

(aba1, aba_pc, aba2, aba3, aba4, aba5, aba_sim, aba_acomp) = st.tabs(
    ["📊 Visão geral", "⚖️ Put/Call", "📅 Por vencimento", "🗓️ Semanal", "📆 Mensal",
     "🚨 Alertas", "🧮 Simulador", "📒 Minhas estratégias"])

with aba1:
    c1, c2 = st.columns(2)
    t = per.groupby(["ativo", "tipo"], as_index=False)["volume"].sum()
    t = t[t.ativo.isin(ativos_rank.head(15).index)]
    grafico(c1, px.bar(t, x="ativo", y="volume", color="tipo", color_discrete_map=CORES,
                       labels=LAB, title="Ativos com mais volume em opções"))
    d = a.groupby(["data", "tipo"], as_index=False)["volume"].sum()
    grafico(c2, px.line(d, x="data", y="volume", color="tipo", color_discrete_map=CORES,
                        labels=LAB, title=f"Volume diário das opções de {ativo}"))
    st.subheader(f"Opções mais negociadas no último pregão ({ultimo:%d/%m/%Y})")
    tabela(st, ranking(df[df.data == ultimo], top))

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
        m1.metric("P/C do último pregão", num(atual_r),
                  f"{atual_r - med:+.2f} vs média".replace(".", ",") if pd.notna(med) else None,
                  delta_color="inverse")
        m2.metric(f"Média de {mm} pregões", num(med) if pd.notna(med) else "–")
        m3.metric("Percentil no período", f"{pct:.0f}%",
                  help="Em quantos % dos pregões do período o P/C ficou abaixo do valor atual.")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d.index, y=d.sup, line=dict(width=0), showlegend=False,
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=d.index, y=d.inf, line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(37,99,235,0.12)", name="Faixa normal",
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=d.index, y=d.ratio, name="Put/Call diário",
                                 line=dict(color="#111827", width=2)))
        fig.add_trace(go.Scatter(x=d.index, y=d.media, name=f"Média {mm}",
                                 line=dict(color="#2563eb", dash="dash")))
        fig.update_layout(title=f"Put/Call ratio diário – {ativo}", hovermode="x unified",
                          yaxis_title="Puts ÷ Calls", yaxis_tickformat=".2f")
        grafico(st, fig, 450)
        if pd.notna(d.sup.iloc[-1]) and atual_r > d.sup.iloc[-1]:
            st.warning("O P/C está **acima da faixa normal**: há mais procura por puts que o "
                       "habitual para este ativo, sinal de proteção ou pessimismo.")
        elif pd.notna(d.inf.iloc[-1]) and atual_r < d.inf.iloc[-1]:
            st.success("O P/C está **abaixo da faixa normal**: calls dominando mais que o "
                       "habitual, sinal de otimismo acima da média.")
        else:
            st.caption("O P/C está dentro da faixa normal do ativo (média ± 2 desvios).")
        r_at = (per.pivot_table(index="ativo", columns="tipo", values=campo, aggfunc="sum")
                  .reindex(columns=["CALL", "PUT"]).fillna(0))
        r_at = r_at.loc[r_at.index.isin(ativos_rank.head(20).index)]
        r_at = r_at[r_at.CALL > 0].assign(ratio=lambda x: x.PUT / x.CALL).sort_values("ratio")
        fb = px.bar(r_at.reset_index(), x="ratio", y="ativo", orientation="h",
                    title="Put/Call no período – 20 ativos mais negociados",
                    labels={"ratio": "Puts ÷ Calls", "ativo": ""})
        fb.update_traces(marker_color=["#2563eb" if x == ativo else "#94a3b8" for x in r_at.index])
        grafico(st, fb, 560)

with aba2:
    vencs = sorted(a.vencimento.dropna().unique())
    if vencs:
        futuros = [v for v in vencs if v >= ultimo]
        padrao = vencs.index(futuros[0]) if futuros else len(vencs) - 1
        v = st.selectbox("Vencimento", vencs, index=padrao,
                         format_func=lambda x: pd.Timestamp(x).strftime("%d/%m/%Y"))
        r = ranking(a[a.vencimento == v], top)
        grafico(st, px.bar(r.sort_values("volume"), x="volume", y="codigo", color="tipo",
                           orientation="h", color_discrete_map=CORES, hover_data=["strike"],
                           labels=LAB, title=f"Opções mais negociadas – vencimento "
                                             f"{pd.Timestamp(v):%d/%m/%Y}"), 600)
        s = a[a.vencimento == v].groupby(["strike", "tipo"], as_index=False)["volume"].sum()
        fs = px.bar(s, x="strike", y="volume", color="tipo", barmode="group",
                    color_discrete_map=CORES, labels=LAB, title="Volume por strike")
        if q:
            fs.add_vline(x=q["preco"], line_dash="dot", line_color="gray",
                         annotation_text=f"{ticker} R$ {num(q['preco'])}")
        grafico(st, fs)
        tabela(st, r)
    else:
        st.info("Sem vencimentos para esse ativo no período.")

for aba, col, nome in [(aba3, "semana", "semanal"), (aba4, "mes", "mensal")]:
    with aba:
        g = a.groupby([col, "tipo"], as_index=False)["volume"].sum()
        grafico(st, px.bar(g, x=col, y="volume", color="tipo", barmode="group",
                           color_discrete_map=CORES, labels=LAB,
                           title=f"Volume {nome} – CALL x PUT"))
        periodos = sorted(a[col].unique(), reverse=True)
        if periodos:
            p = st.selectbox(f"Período {nome}", periodos, key=col,
                             format_func=lambda x, c=col: pd.Timestamp(x).strftime(
                                 "Semana de %d/%m/%Y" if c == "semana" else "%m/%Y"))
            c1, c2 = st.columns(2)
            r = ranking(a[a[col] == p], top)
            c1.markdown("🟢 **CALLs mais negociadas**")
            tabela(c1, r[r.tipo == "CALL"].drop(columns="tipo"))
            c2.markdown("🔴 **PUTs mais negociadas**")
            tabela(c2, r[r.tipo == "PUT"].drop(columns="tipo"))

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
    st.metric(f"Opções com volume anormal em {ultimo:%d/%m/%Y}", len(al))
    st.caption(f"Critério: volume ≥ {num(multiplo, 1)}x a média dos últimos {janela} pregões "
               f"e acima de {brl_c(vol_min)}. Ajuste na barra lateral.")
    if not al.empty:
        grafico(st, px.scatter(al, x="multiplo", y="volume", color="tipo", size="volume",
                               hover_name="codigo", hover_data=["strike", "vencimento"],
                               color_discrete_map=CORES, log_y=True, labels=LAB,
                               title="Volume x quantas vezes acima da média"))
        tabela(st, al)
    else:
        st.success("Nenhum volume anormal com os critérios atuais.")

# ---------- Simulador de estratégias ----------
with aba_sim:
    spot_b3, _ = spot_de(ativo)
    S_ref = float(q["preco"]) if q else float(spot_b3 or 30.0)
    vv_fut = sorted(df[(df.ativo == ativo) & (df.vencimento > ultimo)].vencimento.dropna().unique())
    venc_pad = pd.Timestamp(vv_fut[0]).date() if vv_fut else ultimo.date() + timedelta(days=30)
    dias_pad = int(min(max(np.busday_count(date.today(), venc_pad), 0), 500))

    st.markdown("##### 1. Parâmetros")
    c = st.columns(5)
    S0 = c[0].number_input(f"Preço de {ticker} (R$)", 0.01, 100000.0, round(S_ref, 2), 0.10,
                           key=f"S0_{ticker}_{round(S_ref, 2)}",
                           help=q["fonte"] if q else "Informe o preço manualmente.")
    dias = c[1].number_input("Dias úteis até o vencimento", 0, 500, dias_pad,
                             help=f"Próximo vencimento com opções: {venc_pad:%d/%m/%Y}")
    vol = c[2].number_input("Volatilidade (% a.a.)", 1.0, 300.0, 35.0, 1.0) / 100
    taxa = c[3].number_input("Juros (% a.a.)", 0.0, 50.0, 13.0, 0.25) / 100
    qtd_pad = c[4].number_input("Quantidade", 100, 10_000_000, 100, 100)
    T, r_cont = dias / 252, math.log(1 + taxa)

    st.markdown("##### 2. Estratégia")
    cp0, cp1, cp2, cp3, cp4 = st.columns([1.6, 2.4, 1.3, 1, 1])
    cat = cp0.selectbox("Cenário", CATEGORIAS)
    preset = cp1.selectbox("Estratégia", [k for k, v in ESTRATEGIAS.items() if v[0] == cat])
    passo = cp2.number_input("Distância entre strikes (R$)", 0.01, 10000.0,
                             float(max(round(S0 * 0.05 * 2) / 2, 0.5)), 0.5)
    cp3.write(""); cp4.write("")
    chave = (preset, ticker)
    remontar = cp3.button("Remontar", width="stretch",
                          help="Recalcula as pernas com os parâmetros atuais")
    if st.session_state.get("preset_atual") != chave or remontar:
        st.session_state.preset_atual = chave
        st.session_state.pernas = montar_preset(preset, S0, passo, T, r_cont, vol, qtd_pad, ticker)
        st.session_state.ed_v += 1
    if cp4.button("Limpar", width="stretch"):
        st.session_state.pernas = limpar(pd.DataFrame(columns=COLS_P))
        st.session_state.ed_v += 1
        st.rerun()
    st.info(f"**{preset}:** {ESTRATEGIAS[preset][1]}")

    st.markdown("##### 3. Pernas")
    st.caption("Edite direto na tabela ou adicione linhas. Os prêmios das estratégias prontas "
               "são teóricos (Black-Scholes); use **Adicionar opção real** pra trazer o preço "
               "de mercado.")
    editado = st.data_editor(
        st.session_state.pernas, key=f"ed_{st.session_state.ed_v}", num_rows="dynamic",
        width="stretch", hide_index=True,
        column_config={
            "Operação": st.column_config.SelectboxColumn(options=["Compra", "Venda"],
                                                         required=True, default="Compra"),
            "Tipo": st.column_config.SelectboxColumn(options=["CALL", "PUT", "AÇÃO"],
                                                     required=True, default="CALL"),
            "Código": st.column_config.TextColumn(help="Código da opção ou da ação, usado no "
                                                       "acompanhamento"),
            "Strike": st.column_config.NumberColumn(format="%.2f", min_value=0.0, default=0.0),
            "Prêmio": st.column_config.NumberColumn("Prêmio / preço (R$)", format="%.2f",
                                                    min_value=0.0, default=0.0),
            "Qtd": st.column_config.NumberColumn("Quantidade", min_value=0, step=100,
                                                 default=100, format="localized"),
        })
    pernas = limpar(editado)

    with st.expander("➕ Adicionar opção real (último pregão)"):
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
                                                      f"strike {num(ops.strike[i])} · "
                                                      f"R$ {num(ops.fechamento[i])}")
            op = e3.selectbox("Operação", ["Compra", "Venda"], key="add_op")
            qq = e4.number_input("Quantidade", 1, 10_000_000, int(qtd_pad), 100, key="add_q")
            e5.write("")
            if e5.button("Adicionar", width="stretch"):
                o = ops.iloc[i_op]
                nova = pd.DataFrame([{"Operação": op, "Tipo": o.tipo, "Código": o.codigo,
                                      "Strike": o.strike, "Prêmio": o.fechamento, "Qtd": float(qq)}])
                st.session_state.pernas = limpar(pd.concat([pernas, nova], ignore_index=True))
                st.session_state.ed_v += 1
                st.rerun()
            st.caption(f"Preços do fechamento de {ultimo:%d/%m/%Y}. Faltam "
                       f"{np.busday_count(date.today(), pd.Timestamp(vsel).date())} dias úteis "
                       "para esse vencimento.")

    st.markdown("##### 4. Resultado")
    if pernas.empty:
        st.info("Escolha uma estratégia ou adicione pernas na tabela.")
    else:
        custo, lmax, pmax, bes = resumo(pernas, S0)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Montagem", ("Débito " if custo > 0 else "Crédito ") + brl(abs(custo), 2))
        m2.metric("Lucro máximo", lmax)
        m3.metric("Prejuízo máximo", pmax)
        m4.metric("Break-even", " / ".join(f"R$ {num(b)}" for b in bes) if bes else "–")
        grafico(st, grafico_payoff(pernas, S0, T, r_cont, vol), 480)
        st.caption("Linha azul: resultado no vencimento. Linha tracejada: resultado teórico hoje, "
                   "pela volatilidade e prazo informados. Opções de ações na B3 são americanas; "
                   "o modelo é uma aproximação.")

        s1, s2, s3 = st.columns([2, 1, 1])
        nome_est = s1.text_input("Nome da estratégia", f"{preset} {ticker}")
        venc_est = s2.date_input("Vencimento", venc_pad, key="venc_est", format="DD/MM/YYYY")
        s3.write("")
        if s3.button("💾 Salvar para acompanhar", width="stretch"):
            st.session_state.estrategias.append({
                "id": datetime.now().strftime("%Y%m%d%H%M%S%f"), "nome": nome_est,
                "ativo": ativo, "ticker": ticker, "spot_entrada": S0,
                "criada": date.today().isoformat(), "vencimento": venc_est.isoformat(),
                "pernas": pernas.to_dict("records")})
            salvar_est()
            st.success("Estratégia salva! Veja na aba **📒 Minhas estratégias**.")

# ---------- Acompanhamento ----------
with aba_acomp:
    ests = st.session_state.estrategias
    b1, b2 = st.columns(2)
    b1.download_button("⬇️ Baixar backup das estratégias",
                       json.dumps(ests, ensure_ascii=False, default=float, indent=1),
                       "estrategias.json", "application/json", width="stretch")
    up = b2.file_uploader("Restaurar backup", type="json", label_visibility="collapsed")
    if up is not None and b2.button("Restaurar este backup", width="stretch"):
        st.session_state.estrategias = json.load(up)
        salvar_est()
        st.rerun()
    st.caption("No servidor gratuito, as estratégias podem se perder quando o app reinicia. "
               "Baixe o backup de vez em quando.")

    if not ests:
        st.info("Nenhuma estratégia salva ainda. Monte uma no **🧮 Simulador** e clique em salvar.")
    else:
        linhas, detalhes = [], []
        for e in ests:
            tk = e.get("ticker") or spot_de(e["ativo"])[1] or e["ativo"]
            qe = cotacao(tk)
            p = limpar(pd.DataFrame(e["pernas"]))
            sg = sinal(p)
            atual = p["Código"].map(precos)
            for i in p.index[p["Tipo"] == "AÇÃO"]:
                qa = cotacao(p.at[i, "Código"] or tk)
                if qa:
                    atual[i] = qa["preco"]
            sem_preco = atual.isna()
            atual = atual.fillna(p["Prêmio"])
            p["Preço atual"] = atual
            p["Resultado (R$)"] = sg * p["Qtd"] * (atual - p["Prêmio"])
            custo = float((sg * p["Qtd"] * p["Prêmio"]).sum())
            pl = float(p["Resultado (R$)"].sum())
            try:
                dias_r = int(np.busday_count(date.today(), date.fromisoformat(e["vencimento"])))
            except Exception:
                dias_r = None
            linhas.append({"Estratégia": e["nome"], "Ativo": tk,
                           "Montada em": pd.to_datetime(e["criada"]),
                           "Vencimento": pd.to_datetime(e["vencimento"]), "Dias úteis": dias_r,
                           "Montagem (R$)": custo, "Resultado (R$)": pl,
                           "Resultado (%)": pl / abs(custo) * 100 if custo else None})
            detalhes.append((e, tk, qe, p, sem_preco, pl))

        tab = pd.DataFrame(linhas)
        st.metric("Resultado total das estratégias", brl(tab["Resultado (R$)"].sum(), 2))
        st.dataframe(tab, width="stretch", hide_index=True, column_config={
            "Montada em": st.column_config.DateColumn(format="DD/MM/YYYY"),
            "Vencimento": st.column_config.DateColumn(format="DD/MM/YYYY"),
            "Montagem (R$)": st.column_config.NumberColumn(format="localized"),
            "Resultado (R$)": st.column_config.NumberColumn(format="localized"),
            "Resultado (%)": st.column_config.NumberColumn(format="%.1f%%")})
        st.caption(f"Ações: cotação atual (Yahoo, ~15 min de atraso). Opções: último fechamento "
                   f"da B3 ({ultimo:%d/%m/%Y}); clique em **🔄 Atualizar** na barra lateral "
                   "pra trazer o pregão mais recente.")

        for e, tk, qe, p, sem_preco, pl in detalhes:
            icone = "🟢" if pl >= 0 else "🔴"
            with st.expander(f"{icone} {e['nome']} • {brl(pl, 2)}"):
                if sem_preco.any():
                    st.caption("⚠️ Pernas sem código ou sem negócio recente estão avaliadas pelo "
                               "preço de entrada. Preencha o código pra acompanhar de verdade.")
                st.dataframe(p, width="stretch", hide_index=True, column_config={
                    "Prêmio": st.column_config.NumberColumn("Entrada", format="%.2f"),
                    "Preço atual": st.column_config.NumberColumn(format="%.2f"),
                    "Qtd": st.column_config.NumberColumn("Quantidade", format="localized"),
                    "Resultado (R$)": st.column_config.NumberColumn(format="localized")})
                spot_now = (qe["preco"] if qe else None) or float(e.get("spot_entrada", 0)) or 1.0
                grafico(st, grafico_payoff(p[COLS_P], spot_now, titulo="Payoff no vencimento",
                                           ponto=(spot_now, pl)), 450, key=f"g_{e['id']}")
                if st.button("🗑️ Excluir estratégia", key=f"del_{e['id']}"):
                    st.session_state.estrategias = [x for x in ests if x["id"] != e["id"]]
                    salvar_est()
                    st.rerun()

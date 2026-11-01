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
def ncdf(x):
    """Distribuição normal acumulada (aproximação vetorizada, erro < 1e-7)."""
    x = np.asarray(x, float)
    z = np.abs(x) / math.sqrt(2)
    t = 1 / (1 + 0.3275911 * z)
    y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
             + 0.254829592) * t * np.exp(-z * z)
    return 0.5 * (1 + np.sign(x) * y)

def bs_vec(S, K, T, r, vol, call):
    """Black-Scholes vetorizado. Devolve (preço, d1)."""
    raiz = vol * np.sqrt(T)
    d1 = (np.log(S / K) + (r + vol ** 2 / 2) * T) / raiz
    d2 = d1 - raiz
    desc = np.exp(-r * T)
    c = S * ncdf(d1) - K * desc * ncdf(d2)
    p = K * desc * ncdf(-d2) - S * ncdf(-d1)
    return np.where(call, c, p), d1

def iv_vec(preco, S, K, T, r, call, it=50):
    """Volatilidade implícita por bisseção, para milhares de opções de uma vez."""
    n = len(preco)
    lo, hi = np.full(n, 0.01), np.full(n, 3.0)
    for _ in range(it):
        mid = (lo + hi) / 2
        alto = bs_vec(S, K, T, r, mid, call)[0] > preco
        hi, lo = np.where(alto, mid, hi), np.where(alto, lo, mid)
    vmin = bs_vec(S, K, T, r, np.full(n, 0.011), call)[0]
    vmax = bs_vec(S, K, T, r, np.full(n, 2.99), call)[0]
    return np.where((preco > vmin) & (preco < vmax), (lo + hi) / 2, np.nan)

def gregas(S, K, T, r, vol, call):
    """Delta, gama, theta (R$/dia), vega (R$ por 1 ponto de vol) e prob. de terminar ITM."""
    S, K, T, vol = (np.asarray(x, float) for x in (S, K, T, vol))
    T = np.maximum(T, 1 / 252)
    raiz = vol * np.sqrt(T)
    d1 = (np.log(S / K) + (r + vol ** 2 / 2) * T) / raiz
    d2 = d1 - raiz
    pdf = np.exp(-d1 ** 2 / 2) / math.sqrt(2 * math.pi)
    desc = np.exp(-r * T)
    delta = np.where(call, ncdf(d1), ncdf(d1) - 1)
    gama = pdf / (S * raiz)
    theta_c = -S * pdf * vol / (2 * np.sqrt(T)) - r * K * desc * ncdf(d2)
    theta_p = -S * pdf * vol / (2 * np.sqrt(T)) + r * K * desc * ncdf(-d2)
    theta = np.where(call, theta_c, theta_p) / 252
    vega = S * pdf * np.sqrt(T) / 100
    prob_itm = np.where(call, ncdf(d2), ncdf(-d2))
    return delta, gama, theta, vega, prob_itm

def moneyness(tipo, K, S, tol):
    rel = np.asarray(K, float) / S - 1
    call = np.asarray(tipo) == "CALL"
    return np.where(np.abs(rel) <= tol, "ATM", np.where((rel < 0) == call, "ITM", "OTM"))

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

# ---------- Análises quantitativas ----------
@st.cache_data(show_spinner="Calculando volatilidade implícita...")
def base_quant(versao, raiz, tk, r):
    sp = acoes[acoes.codigo == tk][["data", "fechamento"]].rename(columns={"fechamento": "spot"})
    o = df[df.ativo == raiz][["data", "codigo", "tipo", "strike", "vencimento", "fechamento",
                              "volume", "contratos", "negocios"]].merge(sp, on="data")
    o = o.dropna(subset=["vencimento"])
    o = o[(o.strike > 0) & (o.fechamento > 0) & (o.spot > 0)].copy()
    if o.empty:
        return o, pd.DataFrame()
    o["du"] = np.busday_count(o.data.values.astype("datetime64[D]"),
                              o.vencimento.values.astype("datetime64[D]"))
    o = o[o.du > 0].copy()
    o["T"] = o.du / 252
    call = (o.tipo == "CALL").to_numpy()
    o["iv"] = iv_vec(o.fechamento.to_numpy(), o.spot.to_numpy(), o.strike.to_numpy(),
                     o["T"].to_numpy(), r, call)
    # Série diária de IV: opções perto do dinheiro, 5 a 70 dias úteis, com liquidez
    mon = (o.strike / o.spot - 1).abs()
    b = o[o.iv.between(0.03, 2.5) & (mon <= 0.12) & o.du.between(5, 70) & (o.negocios >= 3)].copy()
    b["w"] = b.volume * np.exp(-((b.strike / b.spot - 1) / 0.04) ** 2)
    b["wiv"] = b.w * b.iv
    def media_pond(x):
        s = x.groupby("data")[["wiv", "w"]].sum()
        return s.wiv / s.w.replace(0, np.nan)
    serie = pd.DataFrame({"IV": media_pond(b), "IV calls": media_pond(b[b.tipo == "CALL"]),
                          "IV puts": media_pond(b[b.tipo == "PUT"])}).dropna(subset=["IV"])
    # Pressão compradora/vendedora estimada: variação do prêmio além do que o delta explica
    iv_dia = serie["IV"].reindex(o.data).to_numpy()
    vol_d = np.where(np.isnan(iv_dia), 0.35, iv_dia)
    d1 = bs_vec(o.spot.to_numpy(), o.strike.to_numpy(), o["T"].to_numpy(), r, vol_d, call)[1]
    o["delta"] = np.where(call, ncdf(d1), ncdf(d1) - 1)
    o = o.sort_values(["codigo", "data"])
    g = o.groupby("codigo")
    resid = (o.fechamento - g.fechamento.shift()) - o.delta * (o.spot - g.spot.shift())
    o["pressao"] = np.sign(resid).fillna(0) * o.volume
    return o, serie

@st.cache_data(ttl=3600, show_spinner=False)
def hist_yahoo(tk):
    try:
        import yfinance as yf
        h = yf.Ticker(f"{tk}.SA").history(period="2y", interval="1d", auto_adjust=False)
        if h.empty or len(h) < 30:
            return None
        h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
        aj = h["Adj Close"] if "Adj Close" in h else h["Close"]
        return h["Close"].astype(float), aj.astype(float)
    except Exception:
        return None

def historico_precos(tk):
    y = hist_yahoo(tk)
    if y is not None:
        return y[0], y[1], "Yahoo Finance, 2 anos"
    x = acoes[acoes.codigo == tk].set_index("data")["fechamento"].sort_index()
    return x, x, "COTAHIST B3"

@st.cache_data(show_spinner="Rodando simulações...")
def monte_carlo(S0, sigma, mu, dias, n, metodo, rets, seed):
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    if metodo == "Bootstrap" and len(rets) > 20:
        base = rets - rets.mean()
        lr = rng.choice(base, size=(n, dias)).astype(np.float32) + np.float32(mu * dt)
    else:
        z = rng.standard_normal((n, dias), dtype=np.float32)
        lr = np.float32((mu - 0.5 * sigma ** 2) * dt) + np.float32(sigma * math.sqrt(dt)) * z
    return (S0 * np.exp(np.cumsum(lr, axis=1))).astype(np.float32)

def max_pain(f):
    """Strike em que o total pago aos titulares das opções seria mínimo (aproximação)."""
    ks = np.sort(f.strike.unique())
    if len(ks) == 0:
        return None
    c = f[f.tipo == "CALL"].groupby("strike").contratos.sum()
    p = f[f.tipo == "PUT"].groupby("strike").contratos.sum()
    custo = [(c * np.maximum(P - c.index, 0)).sum() + (p * np.maximum(p.index - P, 0)).sum()
             for P in ks]
    return float(ks[int(np.argmin(custo))])

# ---------- Montagem automática com opções reais ----------
import itertools

OBJETIVOS = ["Equilibrado", "Maior probabilidade de sucesso", "Melhor valor esperado",
             "Melhor retorno sobre o risco", "Maior crédito recebido"]

def _massas(S, T, r, sigma, G=361):
    """Grade de preços finais e probabilidade de cada um (log-normal)."""
    s = max(sigma, 0.01) * math.sqrt(max(T, 1 / 252))
    z = np.linspace(-5, 5, G)
    x = S * np.exp((r - sigma ** 2 / 2) * max(T, 1 / 252) + s * z)
    cdf = ncdf((z[:-1] + z[1:]) / 2)
    massa = np.diff(np.concatenate([[0.0], cdf, [1.0]]))
    return x, massa

@st.cache_data(show_spinner=False, max_entries=64)
def gerar_montagens(nome, cadeia_v, S, T, r, sigma, qtd, sup, res, min_neg=0,
                    max_strikes=24, faixa=0.25, ancorar=True):
    """Testa todas as combinações de strikes reais que respeitam o formato da estratégia."""
    legs = ESTRATEGIAS[nome][2]
    opc = [lg for lg in legs if lg[1] != "AÇÃO"]
    acao = [lg for lg in legs if lg[1] == "AÇÃO"]
    if not opc or cadeia_v.empty:
        return pd.DataFrame()
    c = cadeia_v[(cadeia_v.negocios >= min_neg) & cadeia_v.strike.between(S * (1 - faixa), S * (1 + faixa))]
    c = c.sort_values("negocios").drop_duplicates(["tipo", "strike"], keep="last")
    livro = {(t, k): row for t, k, row in zip(c.tipo, c.strike, c.itertuples(index=False))}
    strikes = np.array(sorted(c.strike.unique()))
    if len(strikes) > max_strikes:
        strikes = np.sort(strikes[np.argsort(np.abs(strikes - S))[:max_strikes]])
    offs = sorted({lg[2] for lg in opc})
    precisa = {o: {lg[1] for lg in opc if lg[2] == o} for o in offs}
    validos = [{k for k in strikes if all((t, k) in livro for t in precisa[o])} for o in offs]
    if ancorar:  # strikes "abaixo" na estratégia ficam abaixo da cotação, e vice-versa
        validos = [{k for k in v if (o == 0) or (o < 0 and k <= S * 1.01) or (o > 0 and k >= S * 0.99)}
                   for o, v in zip(offs, validos)]
    combos = [cb for cb in itertools.combinations(strikes, len(offs))
              if all(cb[j] in validos[j] for j in range(len(offs)))]
    if not combos:
        return pd.DataFrame()
    combos = np.array(combos, float)                                   # (n, n_offs)
    idx = [offs.index(lg[2]) for lg in opc]
    K = combos[:, idx]                                                 # (n, L)
    is_call = np.array([lg[1] == "CALL" for lg in opc])
    sgq = np.array([(1 if lg[0] == "Compra" else -1) * lg[3] * qtd for lg in opc], float)
    tipos = [lg[1] for lg in opc]
    get = lambda col: np.array([[getattr(livro[(tipos[j], k)], col) for j, k in enumerate(row)]
                                for row in K])
    prem, neg = get("fechamento").astype(float), get("negocios").astype(float)
    cods = get("codigo")
    dlt, tht, vga = (get(cn).astype(float) for cn in ("Delta", "Theta", "Vega"))
    sgq_a = sum((1 if lg[0] == "Compra" else -1) * lg[3] * qtd for lg in acao)

    def payoff(x):                                                     # x: (n, M)
        intr = np.where(is_call[None, :, None], np.maximum(x[:, None, :] - K[:, :, None], 0),
                        np.maximum(K[:, :, None] - x[:, None, :], 0))
        v = (sgq[None, :, None] * (intr - prem[:, :, None])).sum(axis=1)
        return v + sgq_a * (x - S)

    # Lucro e prejuízo máximos exatos: o payoff é linear entre strikes
    kmax = K.max(axis=1, keepdims=True)
    nos = np.concatenate([np.full((len(K), 1), 1e-6), K, kmax * 3, kmax * 6], axis=1)
    pn = payoff(nos)
    inclin = pn[:, -1] - pn[:, -2]
    lucro_max = np.where(inclin > 1e-6, np.inf, pn.max(axis=1))
    perda_max = np.where(inclin < -1e-6, -np.inf, pn.min(axis=1))
    # Probabilidades na grade log-normal
    x, massa = _massas(S, T, r, sigma)
    pop, ev, p50, pmin, pmaxg = [], [], [], [], []
    faixa3 = np.abs(np.linspace(-5, 5, len(x))) <= 3
    for i0 in range(0, len(K), 1500):
        sl = slice(i0, i0 + 1500)
        Ks, ps = K[sl], prem[sl]
        intr = np.where(is_call[None, :, None], np.maximum(x[None, None, :] - Ks[:, :, None], 0),
                        np.maximum(Ks[:, :, None] - x[None, None, :], 0))
        sub = (sgq[None, :, None] * (intr - ps[:, :, None])).sum(axis=1) + sgq_a * (x - S)
        pop.append((sub > 0) @ massa)
        ev.append(sub @ massa)
        lm = lucro_max[sl]
        alvo = np.where(np.isfinite(lm), 0.5 * lm, np.inf)
        p50.append((sub >= np.maximum(alvo, 1e-9)[:, None]) @ massa)
        pmin.append(sub[:, faixa3].min(axis=1))
        pmaxg.append(sub[:, faixa3].max(axis=1))
    pop, ev, p50 = np.concatenate(pop), np.concatenate(ev), np.concatenate(p50)
    p50 = np.where(np.isfinite(lucro_max), p50, np.nan)
    pmin, pmaxg = np.concatenate(pmin), np.concatenate(pmaxg)
    risco = np.where(np.isfinite(perda_max), -perda_max, -pmin)
    custo0 = (sgq[None, :] * prem).sum(axis=1) + sgq_a * S
    risco = np.maximum(risco, np.maximum(0.01 * np.abs(custo0), 1.0))
    potencial = np.where(np.isfinite(lucro_max), lucro_max, pmaxg)
    custo = (sgq[None, :] * prem).sum(axis=1) + sgq_a * S
    # Muros: vendas de call no muro de calls ou acima; vendas de put no muro de puts ou abaixo
    vend_call = (sgq < 0) & is_call
    vend_put = (sgq < 0) & ~is_call
    ok_c = np.all(~vend_call[None, :] | (K >= (res or np.inf) - 1e-9), axis=1) if vend_call.any() else np.ones(len(K), bool)
    ok_p = np.all(~vend_put[None, :] | (K <= (sup or 0) + 1e-9), axis=1) if vend_put.any() else np.ones(len(K), bool)
    tem_venda = vend_call.any() or vend_put.any()
    if res is None:
        ok_c = ok_c & ~vend_call.any()
    if sup is None:
        ok_p = ok_p & ~vend_put.any()
    muro = (ok_c & ok_p) if tem_venda else np.zeros(len(K), bool)
    sg_ = np.sign(sgq)
    desc = [" | ".join(f"{'C' if sg_[j] > 0 else 'V'} {'' if abs(opc[j][3]) == 1 else str(opc[j][3]) + 'x '}"
                       f"{tipos[j]} {num(K[i, j])}" for j in range(len(opc))) for i in range(len(K))]
    if acao:
        desc = ["Ação + " + d for d in desc]
    out = pd.DataFrame({
        "Montagem": desc, "PoP": pop * 100, "P50": p50 * 100, "EV": ev, "Custo": custo,
        "LucroMax": lucro_max, "PerdaMax": perda_max, "Risco": risco, "Potencial": potencial,
        "Delta": (sgq * dlt).sum(axis=1) + sgq_a, "Theta": (sgq * tht).sum(axis=1),
        "Vega": (sgq * vga).sum(axis=1), "MinNeg": neg.min(axis=1), "Muro": muro})
    out["EV/Risco"] = out.EV / out.Risco * 100
    out["Ret/Risco"] = out.Potencial / out.Risco
    out["_K"] = list(K)
    out["_P"] = list(prem)
    out["_C"] = list(cods)
    return out

def pontuar(res, objetivo, ivr, usar_muros):
    if res.empty:
        return res
    r = res.copy()
    rk = lambda s: s.rank(pct=True)
    credito = r.Custo < 0
    ivfit = np.zeros(len(r))
    if ivr is not None and pd.notna(ivr):
        ivfit = np.where((ivr >= 60) & credito | (ivr <= 30) & ~credito, 1.0, 0.0)
    muro = r.Muro.astype(float) if usar_muros else 0.0
    if objetivo == "Maior probabilidade de sucesso":
        sc = r.PoP + 2 * rk(r["EV/Risco"]) + 3 * muro
    elif objetivo == "Melhor valor esperado":
        sc = rk(r["EV/Risco"]) + 0.08 * muro + 0.03 * ivfit
    elif objetivo == "Melhor retorno sobre o risco":
        sc = rk(r["Ret/Risco"]) + 0.15 * rk(r.PoP) + 0.08 * muro
    elif objetivo == "Maior crédito recebido":
        sc = rk(-r.Custo) + 0.2 * rk(r.PoP) + 0.08 * muro
    else:
        sc = (0.40 * rk(r.PoP) + 0.35 * rk(r["EV/Risco"]) + 0.15 * rk(r.MinNeg)
              + 0.10 * muro + 0.05 * ivfit)
    r["Score"] = sc
    return r.sort_values("Score", ascending=False)

def filtrar(res, pop_rng, tipo_m, risco_max, exigir_muro):
    if res.empty:
        return res
    f = res[res.PerdaMax <= 0]          # "lucro garantido" = preço defasado no último negócio
    f = f[f.PoP.between(*pop_rng)]
    if tipo_m == "Só crédito":
        f = f[f.Custo < 0]
    elif tipo_m == "Só débito":
        f = f[f.Custo > 0]
    if risco_max:
        f = f[f.Risco <= risco_max]
    if exigir_muro:
        f = f[f.Muro]
    return f

def pernas_de(nome, linha, qtd, S, cod_acao):
    opc = [lg for lg in ESTRATEGIAS[nome][2] if lg[1] != "AÇÃO"]
    rows = [{"Operação": op, "Tipo": "AÇÃO", "Código": cod_acao, "Strike": 0.0,
             "Prêmio": round(S, 2), "Qtd": float(qtd * m)}
            for op, tp, _, m in ESTRATEGIAS[nome][2] if tp == "AÇÃO"]
    for j, (op, tp, _, m) in enumerate(opc):
        rows.append({"Operação": op, "Tipo": tp, "Código": linha["_C"][j],
                     "Strike": float(linha["_K"][j]), "Prêmio": float(linha["_P"][j]),
                     "Qtd": float(qtd * m)})
    return limpar(pd.DataFrame(rows))

def muros_fluxo(raiz, venc, S, npreg=10):
    """Muro de calls acima e de puts abaixo da cotação, pelos contratos negociados."""
    datas_m = sorted(df.data.unique())[-npreg:]
    f = df[(df.ativo == raiz) & (df.vencimento == venc) & df.data.isin(datas_m)]
    if f.empty:
        return None, None
    pv = (f.pivot_table(index="strike", columns="tipo", values="contratos", aggfunc="sum")
            .reindex(columns=["CALL", "PUT"]).fillna(0))
    ac, ab = pv[pv.index > S], pv[pv.index < S]
    res = float(ac.CALL.idxmax()) if not ac.empty and ac.CALL.max() > 0 else None
    sup = float(ab.PUT.idxmax()) if not ab.empty and ab.PUT.max() > 0 else None
    return sup, res

@st.cache_data(show_spinner=False)
def cadeia(versao, raiz, tk, S, r, vol_pad):
    """Opções do último pregão com IV e gregas."""
    h = df[(df.data == ultimo) & (df.ativo == raiz)]
    if h.empty:
        return h
    c = (h.dropna(subset=["vencimento"])
          .groupby(["codigo", "tipo", "strike", "vencimento"], as_index=False)
          .agg(fechamento=("fechamento", "last"), volume=("volume", "sum"),
               negocios=("negocios", "sum")))
    c = c[(c.strike > 0) & (c.fechamento > 0)].copy()
    sp = acoes[(acoes.codigo == tk) & (acoes.data == ultimo)].fechamento
    S_fech = float(sp.iloc[0]) if len(sp) else S
    du_fech = np.busday_count(np.datetime64(ultimo.date()),
                              c.vencimento.values.astype("datetime64[D]"))
    du_hoje = np.busday_count(np.datetime64(date.today()),
                              c.vencimento.values.astype("datetime64[D]"))
    call = (c.tipo == "CALL").to_numpy()
    T_fech = np.maximum(du_fech, 1) / 252
    c["IV"] = iv_vec(c.fechamento.to_numpy(), np.full(len(c), S_fech),
                     c.strike.to_numpy(), T_fech, r, call)
    vol_use = np.where(np.isnan(c.IV), vol_pad, c.IV)
    c["Dias úteis"] = np.maximum(du_hoje, 0)
    d, g, t, v, p = gregas(np.full(len(c), S), c.strike.to_numpy(),
                           np.maximum(du_hoje, 1) / 252, r, vol_use, call)
    c["Delta"], c["Gama"], c["Theta"], c["Vega"], c["Prob. ITM"] = d, g, t, v, p * 100
    c["|Delta|"] = np.abs(d)
    c["IV"] = c.IV * 100
    return c

def iv_atm(c_v, S):
    """IV no dinheiro do vencimento, ponderada pela proximidade do strike."""
    x = c_v.dropna(subset=["IV"])
    x = x[x.IV.between(3, 250)]
    if x.empty:
        return None
    w = np.exp(-((x.strike / S - 1) / 0.03) ** 2) * np.sqrt(x.negocios + 1)
    return float((w * x.IV).sum() / w.sum() / 100) if w.sum() > 0 else None

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

(aba1, aba_pc, aba2, aba3, aba4, aba5, aba_q, aba_sim, aba_acomp) = st.tabs(
    ["📊 Visão geral", "⚖️ Put/Call", "📅 Por vencimento", "🗓️ Semanal", "📆 Mensal",
     "🚨 Alertas", "🔬 Quant", "🧮 Simulador", "📒 Minhas estratégias"])

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
    if st.session_state.get("_dias_tk") != ticker or "dias_sim" not in st.session_state:
        st.session_state.dias_sim = dias_pad
        st.session_state.venc_est = venc_pad
        st.session_state._dias_tk = ticker
    if "_dias_pend" in st.session_state:
        st.session_state.dias_sim = int(st.session_state.pop("_dias_pend"))
    if "_venc_pend" in st.session_state:
        st.session_state.venc_est = st.session_state.pop("_venc_pend")
    dias = c[1].number_input("Dias úteis até o vencimento", 0, 500, key="dias_sim",
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

    # ===== Montagem automática com opções reais =====
    st.markdown("##### 🤖 Montagem automática com opções reais")
    ch = cadeia(os.path.getmtime(HIST), ativo, ticker, round(S0, 2), round(r_cont, 6), round(vol, 4))
    with st.container(border=True):
        if ch.empty:
            st.caption("Sem opções negociadas no último pregão para montar automaticamente.")
        else:
            vv_a = sorted(ch.vencimento.unique())
            fut_a = [x for x in vv_a if np.busday_count(date.today(), pd.Timestamp(x).date()) >= 1]
            vv_a = fut_a or vv_a
            u1, u2, u3 = st.columns([1.2, 1.8, 1.2])
            v_auto = u1.selectbox("Vencimento", vv_a, key="auto_v",
                                  format_func=lambda x: f"{pd.Timestamp(x):%d/%m/%Y} "
                                  f"({np.busday_count(date.today(), pd.Timestamp(x).date())} d.u.)")
            objetivo = u2.selectbox("Objetivo", OBJETIVOS, key="auto_obj",
                                    help="Equilibrado combina probabilidade, valor esperado, "
                                         "liquidez, muros e IV Rank.")
            tipo_m = u3.selectbox("Montagem", ["Qualquer", "Só crédito", "Só débito"], key="auto_tm")

            c_v = ch[ch.vencimento == v_auto]
            du_a = int(max(np.busday_count(date.today(), pd.Timestamp(v_auto).date()), 1))
            T_a = du_a / 252
            iv_v = iv_atm(c_v, S0)
            _, serie_ctx = base_quant(os.path.getmtime(HIST), ativo, ticker, round(r_cont, 6))
            ivr_ctx = None
            if len(serie_ctx) > 10:
                s_ = serie_ctx["IV"]
                if s_.max() > s_.min():
                    ivr_ctx = float((s_.iloc[-1] - s_.min()) / (s_.max() - s_.min()) * 100)
            sup_a, res_a = muros_fluxo(ativo, v_auto, S0)

            w1, w2, w3, w4, w5 = st.columns(5)
            fonte_sig = w1.selectbox("Vol. para probabilidade",
                                     (["IV do vencimento"] if iv_v else []) + ["Vol. do simulador"],
                                     key="auto_sig",
                                     help="A IV do vencimento é a expectativa de oscilação que o "
                                          "próprio mercado está precificando.")
            sig_a = iv_v if (fonte_sig == "IV do vencimento" and iv_v) else vol
            w2.metric("IV do vencimento", f"{num(iv_v * 100, 1)}%" if iv_v else "–")
            w3.metric("IV Rank", f"{ivr_ctx:.0f}" if ivr_ctx is not None else "–")
            w4.metric("Muro de puts (suporte)", f"R$ {num(sup_a)}" if sup_a else "–")
            w5.metric("Muro de calls (resistência)", f"R$ {num(res_a)}" if res_a else "–")

            if ivr_ctx is not None:
                if ivr_ctx >= 60:
                    st.caption(f"📌 IV Rank {ivr_ctx:.0f}: opções caras. O motor favorece "
                               "montagens de **crédito**; Iron Condor, Jade Lizard, travas de "
                               "crédito e venda coberta tendem a se encaixar melhor.")
                elif ivr_ctx <= 30:
                    st.caption(f"📌 IV Rank {ivr_ctx:.0f}: opções baratas. O motor favorece "
                               "montagens de **débito**; compra de opções, travas de débito, "
                               "borboletas e straddles comprados tendem a se encaixar melhor.")
                else:
                    st.caption(f"📌 IV Rank {ivr_ctx:.0f}: volatilidade em nível intermediário.")

            x1, x2, x3, x4 = st.columns([2.2, 1, 1, 1.3])
            pop_rng = x1.slider("Probabilidade de sucesso (%)", 0, 100, (0, 100), 5,
                                key="auto_pop",
                                help="Chance estimada de terminar com lucro no vencimento. "
                                     "Mais probabilidade costuma significar ganho menor em "
                                     "relação ao risco.")
            min_neg = x2.number_input("Negócios mín./perna", 0, 1000, 3, 1, key="auto_neg",
                                      help="Filtra opções sem liquidez no último pregão.")
            risco_max = x3.number_input("Risco máx. (R$)", 0, 100_000_000, 0, 100, key="auto_rm",
                                        help="0 = sem limite.")
            x4.write("")
            usar_muros = x4.toggle("Priorizar muros", value=True, key="auto_muro",
                                   help="Dá preferência a montagens cujas vendas de call ficam no "
                                        "muro de calls ou acima e as vendas de put no muro de "
                                        "puts ou abaixo.")
            y1, y2 = st.columns(2)
            exigir = y1.toggle("Exigir que todas as vendas fiquem protegidas pelos muros",
                               value=False, key="auto_exig")
            ancorar = y2.toggle("Respeitar a posição dos strikes da estratégia", value=True,
                                key="auto_anc",
                                help="Ex.: no Iron Condor as vendas ficam fora do dinheiro; no "
                                     "financiamento a call vendida fica dentro do dinheiro. "
                                     "Desligue pra testar variações livres.")

            with st.spinner("Testando combinações com as opções reais..."):
                brutos = gerar_montagens(preset, c_v, round(S0, 2), T_a, r_cont, round(sig_a, 4),
                                         int(qtd_pad), sup_a, res_a, int(min_neg),
                                         ancorar=ancorar)
            if brutos.empty:
                st.warning("Não há opções reais suficientes nesse vencimento pra montar essa "
                           "estratégia. Tente outro vencimento ou reduza os negócios mínimos.")
            else:
                cand = pontuar(filtrar(brutos, pop_rng, tipo_m, risco_max, exigir),
                               objetivo, ivr_ctx, usar_muros)
                n_def = int((brutos.PerdaMax > 0).sum())
                st.caption((f"{len(brutos):,} combinações testadas • {len(cand):,} passam nos "
                            f"filtros • probabilidade calculada com vol. de "
                            f"{num(sig_a * 100, 1)}% a.a. e {du_a} dias úteis").replace(",", ".")
                           + (f" • {n_def} descartadas por preço defasado (lucro 'garantido')"
                              if n_def else ""))
                if cand.empty:
                    st.info("Nenhuma montagem passou nos filtros. Amplie a faixa de probabilidade "
                            "ou o risco máximo, ou desligue a exigência dos muros.")
                else:
                    top_n = cand.head(15).reset_index(drop=True)
                    fmt_lim = lambda v: "Ilimitado" if not np.isfinite(v) else v
                    vis = pd.DataFrame({
                        "Montagem": top_n.Montagem,
                        "Prob. sucesso": top_n.PoP,
                        "Prob. ≥50% do lucro": top_n.P50,
                        "Crédito/Débito": np.where(top_n.Custo < 0, "Crédito", "Débito"),
                        "Valor (R$)": top_n.Custo.abs(),
                        "Lucro máx.": [("Ilimitado" if not np.isfinite(v) else brl(v, 0))
                                       for v in top_n.LucroMax],
                        "Prejuízo máx.": [("Ilimitado" if not np.isfinite(v) else brl(v, 0))
                                          for v in top_n.PerdaMax],
                        "Valor esperado": top_n.EV,
                        "Retorno/risco": top_n["Ret/Risco"],
                        "Delta": top_n.Delta, "Theta/dia": top_n.Theta,
                        "Muros": np.where(top_n.Muro, "🛡️", ""),
                        "Liquidez": top_n.MinNeg})
                    sel_a = st.dataframe(
                        vis, width="stretch", hide_index=True, on_select="rerun",
                        selection_mode="single-row",
                        key=f"auto_tab_{preset}_{st.session_state.ed_v}",
                        column_config={
                            "Prob. sucesso": st.column_config.ProgressColumn(
                                min_value=0, max_value=100, format="%.0f%%"),
                            "Prob. ≥50% do lucro": st.column_config.NumberColumn(format="%.0f%%"),
                            "Valor (R$)": st.column_config.NumberColumn(format="%.2f"),
                            "Valor esperado": st.column_config.NumberColumn(format="%.2f"),
                            "Retorno/risco": st.column_config.NumberColumn(format="%.2f"),
                            "Delta": st.column_config.NumberColumn(format="%.0f"),
                            "Theta/dia": st.column_config.NumberColumn(format="%.2f"),
                            "Liquidez": st.column_config.NumberColumn(
                                "Negócios (perna menos líquida)", format="localized")})
                    try:
                        i_sel = list(sel_a.selection.rows)[0]
                    except Exception:
                        i_sel = 0
                    esc = top_n.iloc[i_sel]
                    pern_esc = pernas_de(preset, esc, int(qtd_pad), S0, ticker)
                    pv1, pv2 = st.columns([3, 1.2])
                    figp = grafico_payoff(pern_esc, S0, titulo=f"Prévia: {esc.Montagem}")
                    for nv, cor, nm in [(sup_a, CORES["PUT"], "Muro de puts"),
                                        (res_a, CORES["CALL"], "Muro de calls")]:
                        if nv:
                            figp.add_vline(x=nv, line_color=cor, line_dash="dash",
                                           annotation_text=nm)
                    grafico(pv1, figp, 400, key="auto_prev")
                    pv2.markdown(f"**{'🥇 Melhor montagem' if i_sel == 0 else 'Montagem selecionada'}**")
                    pv2.dataframe(pern_esc[["Operação", "Tipo", "Código", "Strike", "Prêmio"]],
                                  hide_index=True, width="stretch")
                    razoes = []
                    razoes.append(f"Probabilidade de sucesso de **{num(esc.PoP, 0)}%**")
                    if esc.EV > 0:
                        razoes.append("valor esperado **positivo** pelos preços atuais")
                    if esc.Muro:
                        razoes.append("vendas **protegidas pelos muros** de opções")
                    if ivr_ctx is not None and ((ivr_ctx >= 60 and esc.Custo < 0)
                                                or (ivr_ctx <= 30 and esc.Custo > 0)):
                        razoes.append("tipo de montagem **alinhado ao IV Rank**")
                    pv2.markdown("; ".join(razoes) + ".")
                    if pv2.button("✅ Carregar no simulador", type="primary", width="stretch"):
                        st.session_state.pernas = pern_esc
                        st.session_state.ed_v += 1
                        st.session_state._dias_pend = du_a
                        st.session_state._venc_pend = pd.Timestamp(v_auto).date()
                        st.rerun()
                    st.caption("Prêmios do último negócio do pregão, não do livro de ofertas: "
                               "confira os preços na corretora antes de montar. A probabilidade "
                               "assume distribuição log-normal com a volatilidade escolhida; os "
                               "muros entram como preferência, não mudam a probabilidade.")

            with st.expander("📊 Comparar todas as estratégias para o mercado atual"):
                st.caption("Busca a melhor montagem real de cada estratégia nesse vencimento, "
                           "com os mesmos filtros e objetivo acima.")
                if st.button("Rodar comparação", key="auto_cmp"):
                    linhas_c = []
                    barra = st.progress(0.0)
                    for i_e, nm_e in enumerate(PRESETS):
                        b = gerar_montagens(nm_e, c_v, round(S0, 2), T_a, r_cont, round(sig_a, 4),
                                            int(qtd_pad), sup_a, res_a, int(min_neg),
                                            max_strikes=16, ancorar=ancorar)
                        b = pontuar(filtrar(b, pop_rng, tipo_m, risco_max, exigir),
                                    objetivo, ivr_ctx, usar_muros)
                        barra.progress((i_e + 1) / len(PRESETS))
                        if b.empty:
                            continue
                        m_ = b.iloc[0]
                        linhas_c.append({
                            "Cenário": ESTRATEGIAS[nm_e][0], "Estratégia": nm_e,
                            "Melhor montagem": m_.Montagem, "Prob. sucesso": m_.PoP,
                            "Crédito/Débito": "Crédito" if m_.Custo < 0 else "Débito",
                            "Valor esperado": m_.EV, "VE/risco (%)": m_["EV/Risco"],
                            "Retorno/risco": m_["Ret/Risco"],
                            "Prejuízo máx.": "Ilimitado" if not np.isfinite(m_.PerdaMax)
                            else brl(m_.PerdaMax, 0),
                            "Muros": "🛡️" if m_.Muro else "", "_s": m_.Score})
                    barra.empty()
                    st.session_state.auto_cmp_res = pd.DataFrame(linhas_c)
                cmp_res = st.session_state.get("auto_cmp_res")
                if cmp_res is not None and not cmp_res.empty:
                    st.dataframe(
                        cmp_res.sort_values("VE/risco (%)", ascending=False).drop(columns="_s"),
                        width="stretch", hide_index=True, column_config={
                            "Prob. sucesso": st.column_config.ProgressColumn(
                                min_value=0, max_value=100, format="%.0f%%"),
                            "Valor esperado": st.column_config.NumberColumn(format="%.2f"),
                            "VE/risco (%)": st.column_config.NumberColumn(format="%.1f%%"),
                            "Retorno/risco": st.column_config.NumberColumn(format="%.2f")})
                    st.caption("Ordenado pelo valor esperado por real de risco. Pra carregar uma delas, escolha a "
                               "estratégia no seletor acima.")

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

    with st.expander("➕ Adicionar opção real manualmente (último pregão)"):
        ch = cadeia(os.path.getmtime(HIST), ativo, ticker, round(S0, 2), round(r_cont, 6),
                    round(vol, 4))
        if ch.empty:
            st.caption("Esse ativo não teve opções negociadas no último pregão.")
        else:
            vv = sorted(ch.vencimento.unique())
            futuros = [x for x in vv if x >= pd.Timestamp(date.today())]
            f1, f2, f3, f4 = st.columns([1.3, 1.2, 1.8, 1.2])
            vsel = f1.selectbox("Vencimento", vv, key="add_v",
                                index=vv.index(futuros[0]) if futuros else 0,
                                format_func=lambda x: pd.Timestamp(x).strftime("%d/%m/%Y"))
            tipo_sel = f2.segmented_control("Tipo", ["CALL", "PUT"], default="CALL",
                                            key="add_tipo") or "CALL"
            mon_sel = f3.segmented_control("Moneyness", ["ITM", "ATM", "OTM"],
                                           selection_mode="multi",
                                           default=["ITM", "ATM", "OTM"], key="add_mon") or []
            tol = f4.number_input("Faixa ATM (±%)", 0.5, 10.0, 2.0, 0.5, key="add_tol",
                                  help="Strikes a até essa distância da cotação contam como ATM.") / 100
            g1, g2 = st.columns([2, 1])
            dmin, dmax = g1.slider("Delta (em módulo)", 0.0, 1.0, (0.0, 1.0), 0.05,
                                   key="add_delta",
                                   help="Ex.: 0,20 a 0,35 pra opções OTM típicas de venda; "
                                        "perto de 0,50 é o ATM.")
            so_liq = g2.toggle("Só com 5+ negócios", value=False, key="add_liq")

            c = ch[(ch.vencimento == vsel) & (ch.tipo == tipo_sel)].copy()
            c["Moneyness"] = moneyness(c.tipo, c.strike, S0, tol)
            c["Distância (%)"] = (c.strike / S0 - 1) * 100
            c = c[c.Moneyness.isin(mon_sel) & c["|Delta|"].between(dmin, dmax)]
            if so_liq:
                c = c[c.negocios >= 5]
            c = c.sort_values("strike").reset_index(drop=True)

            cor = "🟢" if tipo_sel == "CALL" else "🔴"
            st.caption(f"{cor} {len(c)} {tipo_sel}s • cotação de referência R$ {num(S0)} • "
                       f"{int(c['Dias úteis'].iloc[0]) if len(c) else '–'} dias úteis até o "
                       f"vencimento • selecione uma ou mais linhas na tabela")
            vis = c[["codigo", "Moneyness", "strike", "Distância (%)", "fechamento", "Delta",
                     "|Delta|", "Gama", "Theta", "Vega", "IV", "Prob. ITM", "negocios", "volume"]]
            sel = st.dataframe(
                vis, width="stretch", hide_index=True, key=f"tab_cadeia_{tipo_sel}_{st.session_state.ed_v}",
                on_select="rerun", selection_mode="multi-row", height=380,
                column_config={
                    "codigo": st.column_config.TextColumn("Opção"),
                    "strike": st.column_config.NumberColumn("Strike", format="%.2f"),
                    "Distância (%)": st.column_config.NumberColumn(format="%+.1f%%"),
                    "fechamento": st.column_config.NumberColumn("Prêmio", format="%.2f"),
                    "Delta": st.column_config.NumberColumn(format="%.2f"),
                    "|Delta|": st.column_config.ProgressColumn(
                        "Delta (barra)", min_value=0.0, max_value=1.0, format="%.2f"),
                    "Gama": st.column_config.NumberColumn(format="%.3f"),
                    "Theta": st.column_config.NumberColumn("Theta (R$/dia)", format="%.3f"),
                    "Vega": st.column_config.NumberColumn("Vega (R$/1%)", format="%.3f"),
                    "IV": st.column_config.NumberColumn("IV", format="%.1f%%"),
                    "Prob. ITM": st.column_config.NumberColumn(format="%.0f%%"),
                    "negocios": st.column_config.NumberColumn("Negócios", format="localized"),
                    "volume": st.column_config.NumberColumn("Volume (R$)", format="compact")})
            linhas_sel = []
            try:
                linhas_sel = list(sel.selection.rows)
            except Exception:
                pass

            a1, a2, a3 = st.columns([1, 1, 1.4])
            op = a1.segmented_control("Operação", ["Compra", "Venda"], default="Compra",
                                      key="add_op") or "Compra"
            qq = a2.number_input("Quantidade", 1, 10_000_000, int(qtd_pad), 100, key="add_q")
            a3.write("")
            rotulo = (f"Adicionar {len(linhas_sel)} selecionada(s)" if linhas_sel
                      else "Selecione na tabela")
            if a3.button(rotulo, width="stretch", disabled=not linhas_sel, type="primary"):
                esc = c.iloc[linhas_sel]
                novas = pd.DataFrame({"Operação": op, "Tipo": esc.tipo.values,
                                      "Código": esc.codigo.values, "Strike": esc.strike.values,
                                      "Prêmio": esc.fechamento.values, "Qtd": float(qq)})
                st.session_state.pernas = limpar(pd.concat([pernas, novas], ignore_index=True))
                st.session_state.ed_v += 1
                st.rerun()
            st.caption("Prêmios e IV do fechamento de "
                       f"{ultimo:%d/%m/%Y}. Gregas recalculadas com a cotação de referência e "
                       "o prazo de hoje, usando a IV de cada opção. Delta de call vai de 0 a 1 e "
                       "de put de -1 a 0; em módulo, aproxima a chance de terminar dentro do dinheiro.")

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
        # Gregas da posição: IV de mercado quando a perna tem código, senão a vol informada
        iv_map = (ch.set_index("codigo")["IV"] / 100).to_dict() if not ch.empty else {}
        opc_p = pernas[pernas.Tipo != "AÇÃO"]
        sg_all = sinal(pernas)
        delta_pos = float((sg_all * pernas.Qtd)[(pernas.Tipo == "AÇÃO").to_numpy()].sum())
        gama_pos = theta_pos = vega_pos = 0.0
        if not opc_p.empty:
            vol_p = opc_p["Código"].map(iv_map).astype(float).fillna(vol).to_numpy()
            dg, gg, tg, vg, _ = gregas(np.full(len(opc_p), S0), opc_p.Strike.to_numpy(),
                                       np.full(len(opc_p), max(T, 1 / 252)), r_cont, vol_p,
                                       (opc_p.Tipo == "CALL").to_numpy())
            mult = sinal(opc_p) * opc_p.Qtd.to_numpy()
            delta_pos += float((mult * dg).sum())
            gama_pos, theta_pos, vega_pos = (float((mult * x).sum()) for x in (gg, tg, vg))
        st.markdown("**Gregas da posição**")
        n1, n2, n3, n4 = st.columns(4)
        n1.metric("Delta (ações equivalentes)", num(delta_pos, 0),
                  help="Quantas ações a posição 'vale' agora. Positivo ganha na alta, "
                       "negativo ganha na queda.")
        n2.metric("Exposição em R$ (delta × preço)", brl(delta_pos * S0, 0))
        n3.metric("Theta (R$ por dia)", brl(theta_pos, 2),
                  help="Quanto a posição ganha (+) ou perde (−) por dia só com a passagem do tempo.")
        n4.metric("Vega (R$ por 1 ponto de vol)", brl(vega_pos, 2),
                  help=f"Gama: {num(gama_pos, 2)} ações de delta por R$ 1 de movimento.")
        if abs(delta_pos) < 0.1 * max(pernas.Qtd.max(), 1):
            st.caption("Posição praticamente **delta neutra**: o resultado depende mais de "
                       "tempo e volatilidade do que da direção.")
        else:
            st.caption(f"Posição **{'comprada' if delta_pos > 0 else 'vendida'} em delta**: a "
                       f"cada R$ 1 de {'alta' if delta_pos > 0 else 'queda'} do ativo, ganha "
                       f"cerca de {brl(abs(delta_pos), 0)}.")
        grafico(st, grafico_payoff(pernas, S0, T, r_cont, vol), 480)
        st.caption("Linha azul: resultado no vencimento. Linha tracejada: resultado teórico hoje, "
                   "pela volatilidade e prazo informados. Opções de ações na B3 são americanas; "
                   "o modelo é uma aproximação.")

        s1, s2, s3 = st.columns([2, 1, 1])
        nome_est = s1.text_input("Nome da estratégia", f"{preset} {ticker}")
        venc_est = s2.date_input("Vencimento", key="venc_est", format="DD/MM/YYYY")
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

with aba_q:
    st.caption("Ferramentas estatísticas para estudo. Não são recomendação de investimento: "
               "os níveis e probabilidades são estimativas baseadas no histórico.")
    o_q, serie_iv = base_quant(os.path.getmtime(HIST), ativo, ticker, round(r_cont, 6))
    fech, ajust, fonte_h = historico_precos(ticker)
    rets = np.log(ajust).diff().dropna()
    S_now = float(q["preco"]) if q else (float(fech.iloc[-1]) if len(fech) else None)
    if not S_now:
        st.warning(f"Não encontrei preços de {ticker}. Clique em **📦 Carga completa**.")
        st.stop()

    def hv(h):
        return float(rets.tail(h).std() * math.sqrt(252)) if len(rets) >= h else np.nan

    # ===== 1. IV e IV Rank =====
    st.subheader("1. Volatilidade implícita e IV Rank")
    if serie_iv.empty:
        st.info(f"Para calcular a IV preciso do preço à vista de {ticker} em cada pregão do "
                "histórico. Clique em **📦 Carga completa** na barra lateral.")
        iv_now = np.nan
    else:
        n_iv = len(serie_iv)
        look = (st.slider("Janela do IV Rank (pregões)", 10, n_iv, n_iv) if n_iv > 10 else n_iv)
        s = serie_iv["IV"].tail(look)
        iv_now = float(s.iloc[-1])
        ivr = (iv_now - s.min()) / (s.max() - s.min()) * 100 if s.max() > s.min() else np.nan
        ivp = (s < iv_now).mean() * 100
        hv21 = hv(21)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("IV atual", f"{num(iv_now * 100, 1)}%",
                  help="Média ponderada por volume das opções perto do dinheiro, de 5 a 70 dias úteis.")
        m2.metric("IV Rank", f"{ivr:.0f}" if pd.notna(ivr) else "–",
                  help="Onde a IV atual está entre a mínima (0) e a máxima (100) da janela.")
        m3.metric("IV Percentil", f"{ivp:.0f}%",
                  help="Em quantos % dos pregões da janela a IV ficou abaixo da atual.")
        m4.metric("IV ÷ Vol. histórica 30d", num(iv_now / hv21) if hv21 else "–",
                  help="Acima de 1: o mercado precifica mais oscilação do que a ação vem tendo.")
        if pd.notna(ivr):
            if ivr >= 70:
                st.warning(f"IV Rank alto ({ivr:.0f}): opções **caras** em relação ao histórico. "
                           "Costuma favorecer estratégias vendidas, como Iron Condor, venda "
                           "coberta e travas de crédito.")
            elif ivr <= 30:
                st.success(f"IV Rank baixo ({ivr:.0f}): opções **baratas** em relação ao histórico. "
                           "Costuma favorecer compra de opções, travas de débito e straddles.")
            else:
                st.caption(f"IV Rank intermediário ({ivr:.0f}).")
        hv_roll = (rets.rolling(21).std() * math.sqrt(252)).reindex(serie_iv.index)
        fiv = go.Figure()
        for col, cor, traco in [("IV", "#111827", None), ("IV calls", CORES["CALL"], "dot"),
                                ("IV puts", CORES["PUT"], "dot")]:
            fiv.add_trace(go.Scatter(x=serie_iv.index, y=serie_iv[col] * 100, name=col,
                                     line=dict(color=cor, dash=traco, width=2 if col == "IV" else 1)))
        fiv.add_trace(go.Scatter(x=hv_roll.index, y=hv_roll * 100, name="Vol. histórica 30d",
                                 line=dict(color="#2563eb", dash="dash")))
        fiv.update_layout(title=f"Volatilidade implícita x histórica – {ticker}",
                          yaxis_title="% ao ano", hovermode="x unified", yaxis_tickformat=".1f")
        grafico(st, fiv)
        if n_iv < 200:
            st.caption(f"O IV Rank clássico usa 1 ano (~252 pregões). Aqui há {n_iv} pregões de "
                       "opções carregados; aumente **Meses de histórico** pra uma leitura mais "
                       "robusta.")

    # ===== 2. Desvio padrão por horizonte =====
    st.subheader("2. Desvio padrão e volatilidade histórica")
    JANELAS = {"1 semana": 5, "30 dias": 21, "3 meses": 63, "6 meses": 126, "1 ano": 252}
    j1, j2 = st.columns([3, 1])
    sel = j1.multiselect("Horizontes", list(JANELAS) + ["Personalizado"],
                         default=["1 semana", "30 dias", "3 meses"])
    hz = {k: JANELAS[k] for k in sel if k in JANELAS}
    if "Personalizado" in sel:
        hpers = j2.number_input("Pregões (personalizado)", 2, 500, 42)
        hz[f"{hpers} pregões"] = int(hpers)
    linhas_dp = []
    for nome_h, h in hz.items():
        if len(fech) <= h:
            continue
        vol_a = hv(h)
        janela_px = fech.tail(h)
        mov = S_now * vol_a * math.sqrt(h / 252)
        linhas_dp.append({
            "Horizonte": nome_h, "Pregões": h, "Vol. histórica (% a.a.)": vol_a * 100,
            "Desvio padrão dos fechamentos (R$)": float(janela_px.std()),
            "Média dos fechamentos (R$)": float(janela_px.mean()),
            "Retorno no período (%)": float((fech.iloc[-1] / fech.iloc[-h - 1] - 1) * 100),
            "Movimento esperado 1σ (R$)": mov,
            "Faixa 1σ (68%)": f"R$ {num(S_now - mov)} a R$ {num(S_now + mov)}",
            "Faixa 2σ (95%)": f"R$ {num(max(S_now - 2 * mov, 0))} a R$ {num(S_now + 2 * mov)}"})
    if linhas_dp:
        st.dataframe(pd.DataFrame(linhas_dp), width="stretch", hide_index=True, column_config={
            "Vol. histórica (% a.a.)": st.column_config.NumberColumn(format="%.1f%%"),
            "Desvio padrão dos fechamentos (R$)": st.column_config.NumberColumn(format="%.2f"),
            "Média dos fechamentos (R$)": st.column_config.NumberColumn(format="%.2f"),
            "Retorno no período (%)": st.column_config.NumberColumn(format="%+.1f%%"),
            "Movimento esperado 1σ (R$)": st.column_config.NumberColumn(format="%.2f")})
        st.caption(f"Preços: {fonte_h}. O movimento esperado projeta a volatilidade histórica "
                   "do horizonte a partir da cotação atual.")
        h_graf = st.radio("Bandas no gráfico", list(hz), horizontal=True, index=min(1, len(hz) - 1))
        h = hz[h_graf]
        base_px = fech.tail(252 + h)
        mmv, dpv = base_px.rolling(h).mean(), base_px.rolling(h).std()
        base_px, mmv, dpv = base_px.tail(252), mmv.tail(252), dpv.tail(252)
        fdp = go.Figure()
        for k, alfa in [(2, 0.08), (1, 0.16)]:
            fdp.add_trace(go.Scatter(x=mmv.index, y=mmv + k * dpv, line=dict(width=0),
                                     showlegend=False, hoverinfo="skip"))
            fdp.add_trace(go.Scatter(x=mmv.index, y=mmv - k * dpv, line=dict(width=0),
                                     fill="tonexty", fillcolor=f"rgba(37,99,235,{alfa})",
                                     name=f"Média ± {k}σ", hoverinfo="skip"))
        fdp.add_trace(go.Scatter(x=base_px.index, y=base_px, name="Fechamento",
                                 line=dict(color="#111827", width=2)))
        fdp.add_trace(go.Scatter(x=mmv.index, y=mmv, name=f"Média {h}",
                                 line=dict(color="#2563eb", dash="dash")))
        # Cone de projeção para o horizonte escolhido
        vol_h = hv(h)
        futuro = pd.bdate_range(base_px.index[-1], periods=h + 1)
        tt = np.arange(h + 1) / 252
        for k, cor in [(1, "rgba(245,158,11,0.9)"), (2, "rgba(245,158,11,0.5)")]:
            for sg in (1, -1):
                fdp.add_trace(go.Scatter(x=futuro, y=S_now * np.exp(sg * k * vol_h * np.sqrt(tt)),
                                         line=dict(color=cor, dash="dot"), hoverinfo="skip",
                                         name=f"Projeção ±{k}σ", showlegend=sg == 1))
        fdp.update_layout(title=f"{ticker} – fechamentos com bandas de desvio padrão ({h_graf})",
                          yaxis_title="R$", hovermode="x unified", yaxis_tickformat=",.2f")
        grafico(st, fdp, 480)
    else:
        st.info("Histórico de preços insuficiente para os horizontes escolhidos.")

    # ===== 3. Suportes e resistências pelo fluxo de opções =====
    st.subheader("3. Suportes e resistências pelo fluxo de opções")
    res_wall = sup_wall = mp = None
    if o_q.empty:
        st.info("Sem dados de opções com preço à vista para esse ativo.")
    else:
        vencs_q = sorted(o_q[o_q.vencimento >= ultimo].vencimento.unique())
        f1, f2 = st.columns([2, 1])
        vsel_q = f1.multiselect("Vencimentos", vencs_q, default=vencs_q[:1],
                                format_func=lambda x: pd.Timestamp(x).strftime("%d/%m/%Y"))
        npreg = f2.slider("Negociações dos últimos N pregões", 1, 60, 10)
        datas_q = sorted(o_q.data.unique())[-npreg:]
        f = o_q[o_q.vencimento.isin(vsel_q) & o_q.data.isin(datas_q)]
        f = f[(f.strike > S_now * 0.6) & (f.strike < S_now * 1.4)]
        if f.empty:
            st.info("Sem negócios nesses vencimentos e período.")
        else:
            pv = (f.pivot_table(index="strike", columns="tipo", values="contratos", aggfunc="sum")
                    .reindex(columns=["CALL", "PUT"]).fillna(0))
            pr = (f.pivot_table(index="strike", columns="tipo", values="pressao", aggfunc="sum")
                    .reindex(columns=["CALL", "PUT"]).fillna(0))
            acima, abaixo = pv[pv.index > S_now], pv[pv.index < S_now]
            res_wall = float(acima.CALL.idxmax()) if not acima.empty and acima.CALL.max() > 0 else None
            sup_wall = float(abaixo.PUT.idxmax()) if not abaixo.empty and abaixo.PUT.max() > 0 else None
            mp = max_pain(f)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Cotação", f"R$ {num(S_now)}")
            m2.metric("Resistência (muro de calls)", f"R$ {num(res_wall)}" if res_wall else "–",
                      f"{num((res_wall / S_now - 1) * 100, 1)}%" if res_wall else None,
                      help="Strike acima da cotação com mais contratos de call negociados.")
            m3.metric("Suporte (muro de puts)", f"R$ {num(sup_wall)}" if sup_wall else "–",
                      f"{num((sup_wall / S_now - 1) * 100, 1)}%" if sup_wall else None,
                      help="Strike abaixo da cotação com mais contratos de put negociados.")
            m4.metric("Max pain (aprox.)", f"R$ {num(mp)}" if mp else "–",
                      help="Preço no vencimento em que os titulares das opções receberiam menos. "
                           "Aproximado pelos contratos negociados, já que a B3 não traz o "
                           "contrato em aberto no arquivo histórico.")

            fw = go.Figure()
            fw.add_trace(go.Bar(y=pv.index, x=pv.CALL, orientation="h", name="Calls",
                                marker_color=CORES["CALL"]))
            fw.add_trace(go.Bar(y=pv.index, x=-pv.PUT, orientation="h", name="Puts",
                                marker_color=CORES["PUT"], customdata=pv.PUT,
                                hovertemplate="Strike %{y}<br>Puts: %{customdata:,.0f}<extra></extra>"))
            fw.add_hline(y=S_now, line_color="#111827", line_dash="dot",
                         annotation_text=f"Cotação {num(S_now)}")
            if res_wall:
                fw.add_hline(y=res_wall, line_color=CORES["CALL"], annotation_text="Resistência")
            if sup_wall:
                fw.add_hline(y=sup_wall, line_color=CORES["PUT"], annotation_text="Suporte")
            fw.update_layout(barmode="relative", title="Contratos negociados por strike "
                             "(puts à esquerda, calls à direita)", yaxis_title="Strike (R$)",
                             xaxis_title="Contratos")
            grafico(st, fw, 600)

            fpz = go.Figure()
            for tp in ["CALL", "PUT"]:
                fpz.add_trace(go.Bar(x=pr.index, y=pr[tp], name=f"{tp}s",
                                     marker_color=CORES[tp]))
            fpz.add_vline(x=S_now, line_dash="dot", line_color="#111827")
            fpz.update_layout(barmode="group", title="Pressão estimada: acima de zero = mais "
                              "compra, abaixo de zero = mais venda", xaxis_title="Strike (R$)",
                              yaxis_title="Volume líquido estimado (R$)")
            grafico(st, fpz)

            leitura = []
            if res_wall:
                pc = pr.CALL.get(res_wall, 0)
                leitura.append(f"**Resistência em R$ {num(res_wall)}:** " + (
                    "predomínio estimado de **venda de calls**. Vendedores apostam que o preço "
                    "não passa desse nível, o que reforça a resistência." if pc < 0 else
                    "predomínio estimado de **compra de calls**. Há aposta em rompimento, então "
                    "a resistência pode ser mais fraca."))
            if sup_wall:
                pp = pr.PUT.get(sup_wall, 0)
                leitura.append(f"**Suporte em R$ {num(sup_wall)}:** " + (
                    "predomínio estimado de **venda de puts**. Vendedores aceitam comprar a ação "
                    "nesse preço, o que reforça o suporte." if pp < 0 else
                    "predomínio estimado de **compra de puts**. É procura por proteção nesse "
                    "nível; se for rompido, pode acelerar a queda."))
            for t_ in leitura:
                st.markdown(t_)
            niveis = pd.DataFrame({
                "Strike": pv.index, "Calls (contratos)": pv.CALL.values,
                "Puts (contratos)": pv.PUT.values,
                "Pressão calls (R$)": pr.CALL.reindex(pv.index).values,
                "Pressão puts (R$)": pr.PUT.reindex(pv.index).values})
            niveis["Total"] = niveis["Calls (contratos)"] + niveis["Puts (contratos)"]
            with st.expander("Ver tabela completa por strike"):
                st.dataframe(niveis.sort_values("Total", ascending=False), width="stretch",
                             hide_index=True, column_config={
                                 "Strike": st.column_config.NumberColumn(format="%.2f"),
                                 "Calls (contratos)": st.column_config.NumberColumn(format="localized"),
                                 "Puts (contratos)": st.column_config.NumberColumn(format="localized"),
                                 "Total": st.column_config.NumberColumn(format="localized"),
                                 "Pressão calls (R$)": st.column_config.NumberColumn(format="compact"),
                                 "Pressão puts (R$)": st.column_config.NumberColumn(format="compact")})
            st.caption("Como a pressão é estimada: para cada opção e pregão, comparo a variação do "
                       "prêmio com o que o delta explicaria pela variação da ação. Se o prêmio subiu "
                       "mais que o esperado, conto o volume como compra; se subiu menos, como venda. "
                       "O arquivo da B3 não informa quem foi o agressor, então é uma aproximação.")

    # ===== 4. Monte Carlo =====
    st.subheader("4. Simulação de Monte Carlo")
    c = st.columns(4)
    dias_mc = c[0].number_input("Horizonte (dias úteis)", 1, 252, int(max(dias, 1)),
                                help="Por padrão, o prazo do simulador de estratégias.")
    n_sim = c[1].select_slider("Simulações", [1000, 5000, 10000, 20000], 10000)
    metodo = c[2].selectbox("Modelo", ["Normal (GBM)", "Bootstrap"],
                            help="Bootstrap sorteia retornos diários reais do histórico, o que "
                                 "preserva caudas gordas e dias extremos.")
    seed = c[3].number_input("Semente", 0, 999999, 42, help="Mesma semente = mesmo resultado.")
    c2 = st.columns(4)
    fontes_vol = {"IV atual": iv_now, "Vol. histórica 30 dias": hv(21),
                  "Vol. histórica 3 meses": hv(63), "Vol. histórica 1 ano": hv(252)}
    opcoes_vol = [k for k, v in fontes_vol.items() if pd.notna(v)] + ["Manual"]
    fonte_vol = c2[0].selectbox("Volatilidade", opcoes_vol, disabled=metodo == "Bootstrap")
    vol_manual = c2[1].number_input("Vol. manual (% a.a.)", 1.0, 300.0, 35.0, 1.0,
                                    disabled=fonte_vol != "Manual")
    sigma = vol_manual / 100 if fonte_vol == "Manual" else float(fontes_vol[fonte_vol])
    tend = c2[2].selectbox("Tendência", ["Neutra (taxa de juros)", "Zero", "Histórica (1 ano)"])
    mu = {"Neutra (taxa de juros)": r_cont, "Zero": 0.0,
          "Histórica (1 ano)": float(rets.tail(252).mean() * 252) if len(rets) else 0.0}[tend]
    extras_txt = c2[3].text_input("Níveis extras (R$)", "",
                                  help="Preços separados por ponto e vírgula, ex.: 30; 35,5")

    caminhos = monte_carlo(round(S_now, 4), round(sigma, 6), round(mu, 6), int(dias_mc),
                           int(n_sim), "Bootstrap" if metodo == "Bootstrap" else "GBM",
                           rets.to_numpy(dtype=float), int(seed))
    final = caminhos[:, -1].astype(float)
    st.caption(f"{n_sim:,} simulações • {dias_mc} dias úteis • ".replace(",", ".") +
               (f"volatilidade {num(sigma * 100, 1)}% a.a." if metodo != "Bootstrap"
                else f"retornos sorteados do histórico ({fonte_h})") +
               f" • tendência {num(mu * 100, 1)}% a.a.")

    pcts = [5, 25, 50, 75, 95]
    faixas = np.percentile(caminhos, pcts, axis=0)
    eixo = pd.bdate_range(pd.Timestamp(date.today()), periods=dias_mc + 1)[1:]
    ffan = go.Figure()
    for i_c in range(min(40, n_sim)):
        ffan.add_trace(go.Scatter(x=eixo, y=caminhos[i_c], line=dict(width=0.6, color="rgba(148,163,184,0.35)"),
                                  showlegend=False, hoverinfo="skip"))
    for (lo_i, hi_i), alfa, nm in [((0, 4), 0.15, "90% dos cenários"), ((1, 3), 0.3, "50% dos cenários")]:
        ffan.add_trace(go.Scatter(x=eixo, y=faixas[hi_i], line=dict(width=0), showlegend=False,
                                  hoverinfo="skip"))
        ffan.add_trace(go.Scatter(x=eixo, y=faixas[lo_i], line=dict(width=0), fill="tonexty",
                                  fillcolor=f"rgba(37,99,235,{alfa})", name=nm))
    ffan.add_trace(go.Scatter(x=eixo, y=faixas[2], name="Mediana", line=dict(color="#2563eb", width=2)))
    for nv, cor, nm in [(res_wall, CORES["CALL"], "Resistência"), (sup_wall, CORES["PUT"], "Suporte")]:
        if nv:
            ffan.add_hline(y=nv, line_color=cor, line_dash="dash", annotation_text=nm)
    ffan.update_layout(title="Leque de cenários simulados", yaxis_title="R$",
                       hovermode="x unified", yaxis_tickformat=",.2f")
    g1, g2 = st.columns(2)
    grafico(g1, ffan)
    fh = go.Figure(go.Histogram(x=final, nbinsx=80, marker_color="#94a3b8", name="Cenários"))
    fh.add_vline(x=S_now, line_color="#111827", line_dash="dot", annotation_text="Hoje")
    for nv, cor in [(res_wall, CORES["CALL"]), (sup_wall, CORES["PUT"])]:
        if nv:
            fh.add_vline(x=nv, line_color=cor, line_dash="dash")
    fh.update_layout(title=f"Distribuição do preço em {dias_mc} dias úteis", xaxis_title="R$",
                     yaxis_title="Cenários", showlegend=False, xaxis_tickformat=",.2f")
    grafico(g2, fh)

    mov1 = S_now * sigma * math.sqrt(dias_mc / 252)
    niveis_mc = [("Suporte (muro de puts)", sup_wall), ("Resistência (muro de calls)", res_wall),
                 ("Max pain", mp), ("-1σ", S_now - mov1), ("+1σ", S_now + mov1),
                 ("-2σ", S_now - 2 * mov1), ("+2σ", S_now + 2 * mov1)]
    for parte in extras_txt.split(";"):
        try:
            niveis_mc.append(("Nível extra", float(parte.strip().replace(",", "."))))
        except ValueError:
            pass
    maxs, mins = caminhos.max(axis=1), caminhos.min(axis=1)
    prob = []
    for nm, nv in niveis_mc:
        if not nv or nv <= 0:
            continue
        toca = (maxs >= nv) if nv >= S_now else (mins <= nv)
        prob.append({"Nível": nm, "Preço (R$)": nv, "Distância (%)": (nv / S_now - 1) * 100,
                     "Fechar acima (%)": (final > nv).mean() * 100,
                     "Fechar abaixo (%)": (final < nv).mean() * 100,
                     "Tocar no caminho (%)": toca.mean() * 100})
    t1, t2 = st.columns([3, 1])
    t1.markdown("**Probabilidades por nível**")
    t1.dataframe(pd.DataFrame(prob).sort_values("Preço (R$)"), width="stretch", hide_index=True,
                 column_config={
                     "Preço (R$)": st.column_config.NumberColumn(format="%.2f"),
                     "Distância (%)": st.column_config.NumberColumn(format="%+.1f%%"),
                     "Fechar acima (%)": st.column_config.NumberColumn(format="%.1f%%"),
                     "Fechar abaixo (%)": st.column_config.NumberColumn(format="%.1f%%"),
                     "Tocar no caminho (%)": st.column_config.NumberColumn(format="%.1f%%")})
    t2.markdown("**Faixas de preço final**")
    t2.dataframe(pd.DataFrame({"Percentil": [f"{p_}%" for p_ in pcts],
                               "Preço (R$)": np.percentile(final, pcts)}),
                 width="stretch", hide_index=True,
                 column_config={"Preço (R$)": st.column_config.NumberColumn(format="%.2f")})

    if not pernas.empty:
        st.markdown(f"**Estratégia do simulador no Monte Carlo:** {preset}")
        pl_mc = resultado(final, pernas)
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Probabilidade de lucro", f"{num((pl_mc > 0).mean() * 100, 1)}%")
        e2.metric("Resultado médio", brl(pl_mc.mean(), 2))
        e3.metric("Pior 5% (VaR)", brl(np.percentile(pl_mc, 5), 2))
        e4.metric("Melhor 5%", brl(np.percentile(pl_mc, 95), 2))
        fpl = go.Figure(go.Histogram(x=pl_mc, nbinsx=80, marker_color="#2563eb"))
        fpl.add_vline(x=0, line_color="#111827")
        fpl.update_layout(title="Distribuição do resultado da estratégia no vencimento",
                          xaxis_title="Resultado (R$)", yaxis_title="Cenários", showlegend=False,
                          xaxis_tickformat=",.0f")
        grafico(st, fpl)
        if dias_mc != dias:
            st.caption(f"O horizonte da simulação ({dias_mc} d.u.) é diferente do prazo do "
                       f"simulador ({dias} d.u.). O resultado considera o payoff no vencimento.")
    else:
        st.caption("Monte uma estratégia no **🧮 Simulador** pra ver a probabilidade de lucro dela aqui.")

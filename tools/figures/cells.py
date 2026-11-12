# %% [md]
# # Figuras e tabelas do artigo
#
# Este notebook lê os resultados que já estão no seu Drive (`qorder_results`) e gera as figuras em PDF (vetorial) e as tabelas em texto.
# Não treina nada: roda em ~1 minuto.
#
# **As escolhas visuais são suas.** Tudo o que define a aparência está na célula *Estilo* (fonte, tamanho, cores, largura, o que entra em cada figura). Mude à vontade e rode de novo.
#
# Ordem: 1. Drive → 2. Estilo → 3. Dados → 4 em diante: uma célula por figura ou tabela. Os arquivos vão para `qorder_results/figuras/`.

# %% [md]
# ## 1. Drive

# %%
import os
from google.colab import drive
drive.mount('/content/drive')
RESULTS = '/content/drive/MyDrive/qorder_results'
OUT = f'{RESULTS}/figuras'
os.makedirs(OUT, exist_ok=True)
print('Lendo de', RESULTS, '| salvando em', OUT)

# %% [md]
# ## 2. Estilo
# O modelo IEEE usa Times. `STIXGeneral` é uma fonte equivalente que já vem no matplotlib.
# Uma coluna IEEE tem ~3,5 polegadas de largura; as duas colunas juntas, ~7,16.

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt

FONTE = 'STIXGeneral'      # fonte do texto das figuras
TAMANHO = 8                # pt, na largura final
COLUNA = 3.5               # polegadas (uma coluna IEEE)
LINHA = 0.9                # espessura das linhas
MARCADOR = 3               # tamanho dos marcadores

# Uma cor fixa por ordenação, igual em todas as figuras
CORES = {
    'heuristic':      '#1b6ca8',
    'default':        '#7f7f7f',
    'reversed':       '#d08c1f',
    'random':         '#5a9e4b',
    'anti_heuristic': '#b23a48',
}
NOMES = {  # rótulos que aparecem nas figuras
    'heuristic': 'Heuristic', 'default': 'Default', 'reversed': 'Reversed',
    'random': 'Random', 'anti_heuristic': 'Anti-heuristic',
}

mpl.rcParams.update({
    'font.family': FONTE, 'mathtext.fontset': 'stix', 'font.size': TAMANHO,
    'axes.labelsize': TAMANHO, 'xtick.labelsize': TAMANHO, 'ytick.labelsize': TAMANHO,
    'legend.fontsize': TAMANHO - 1, 'axes.linewidth': 0.6, 'lines.linewidth': LINHA,
    'lines.markersize': MARCADOR, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'axes.grid': False, 'legend.frameon': False, 'axes.spines.top': False,
    'axes.spines.right': False, 'pdf.fonttype': 42, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})

# Nomes dos datasets como aparecem no texto (usados na Fig. 4 e na Tabela II)
NOMES_DS = {
    'breast_cancer': 'Breast Cancer', 'diabetes': 'Diabetes',
    'fashion_shuffled': 'Fashion-MNIST (shuffled)', 'ionosphere': 'Ionosphere',
    'magic': 'MAGIC', 'mnist_shuffled': 'MNIST (shuffled)',
    'qsar_biodeg': 'QSAR biodegradation', 'spambase': 'Spambase', 'wine': 'Wine',
}

def salvar(fig, nome):
    fig.savefig(f'{OUT}/{nome}.pdf')
    fig.savefig(f'{OUT}/{nome}.png', dpi=300)
    print('salvo:', f'{OUT}/{nome}.pdf')

# %% [md]
# ## 3. Dados

# %%
import glob
import numpy as np
import pandas as pd
from scipy import stats

depth = pd.read_csv(f'{RESULTS}/exp3_depth_cost/runs.csv')          # breast cancer, L = 1..7, custo local e global
grads = pd.read_csv(f'{RESULTS}/exp4_gradients/gradients.csv')     # variância do gradiente
resplit = {os.path.basename(d).removeprefix('exp8_').removesuffix('_resplit'): pd.read_csv(f'{d}/runs.csv')
           for d in sorted(glob.glob(f'{RESULTS}/exp8_*_resplit')) if os.path.exists(f'{d}/runs.csv')}
print('Profundidade:', sorted(depth.m_n_layers.unique()), '| datasets com divisão variável:', list(resplit))

def media_ic(x, n_boot=2000, seed=0):
    """Média e intervalo de confiança de 95% por bootstrap."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    b = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return x.mean(), *np.percentile(b, [2.5, 97.5])

def diff_ic(a, b, n_boot=2000, seed=0):
    """Diferença de médias (a - b) com IC 95% por bootstrap (amostras independentes)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    d = rng.choice(a, (n_boot, len(a))).mean(1) - rng.choice(b, (n_boot, len(b))).mean(1)
    return a.mean() - b.mean(), *np.percentile(d, [2.5, 97.5])

# %% [md]
# ## 4. Fig. 1: circuito e cone causal
# Circuito de 8 qubits com L camadas. Cada camada: RY em todos os qubits e CZ entre vizinhos em dois sub-passos (pares 0–1, 2–3, … e depois 1–2, 3–4, …), desenhados em posições diferentes para que cada CZ apareça só entre os seus dois qubits.
# A região sombreada é o cone causal de ⟨Z₀⟩, calculado a partir do circuito (não desenhado à mão): voltando do fim, o suporte cresce um vizinho a cada CZ que toca um qubit já no cone.

# %%
N_QUBITS = 8
N_CAMADAS = 2
QUBIT_MEDIDO = 0
COR_CONE = '#dbe8f4'
ALTURA = 2.3   # polegadas

def pares(sub):  # sub-passo 0: (0,1),(2,3)...; sub-passo 1: (1,2),(3,4)...
    return [(q, q + 1) for q in range(sub, N_QUBITS - 1, 2)]

# sequência de colunas do circuito: ('ry', camada) ou ('cz', sub-passo)
colunas = []
for l in range(N_CAMADAS):
    colunas += [('ry', l), ('cz', 0), ('cz', 1)]
colunas += [('ry', N_CAMADAS)]

# suporte do observável em cada coluna, calculado de trás para frente.
# As CZ são diagonais: um qubit só "espalha" para o vizinho se já carregava uma parte X/Y,
# isto é, se estava no suporte antes deste bloco de CZ (depois do R_Y seguinte).
# Qubits que entram no cone dentro de um bloco carregam só Z e não espalham no mesmo bloco.
suporte = [None] * len(colunas)
atual = {QUBIT_MEDIDO}
base = None
for i in range(len(colunas) - 1, -1, -1):
    tipo, k = colunas[i]
    if tipo == 'cz':
        if base is None:
            base = set(atual)          # suporte que chega ao bloco de CZ vindo da direita
        for a, b in pares(k):
            if a in base or b in base:
                atual |= {a, b}
    else:
        base = None
    suporte[i] = set(atual)
print('qubits no cone:', sorted(suporte[0]), f'({len(suporte[0])} = L + 1)')

fig, ax = plt.subplots(figsize=(COLUNA, ALTURA))
x = np.arange(len(colunas)) * 1.0
y = lambda q: N_QUBITS - 1 - q
for i, (tipo, k) in enumerate(colunas):           # sombra do cone, coluna a coluna
    for q in suporte[i]:
        ax.add_patch(plt.Rectangle((x[i] - 0.5, y(q) - 0.5), 1.0, 1.0, color=COR_CONE, lw=0, zorder=0))
for q in range(N_QUBITS):                          # fios
    ax.plot([x[0] - 0.8, x[-1] + 1.0], [y(q)] * 2, color='black', lw=0.6, zorder=1)
    ax.text(x[0] - 0.95, y(q), f'$q_{q}$', ha='right', va='center')
for i, (tipo, k) in enumerate(colunas):            # portas
    if tipo == 'ry':
        for q in range(N_QUBITS):
            ax.add_patch(plt.Rectangle((x[i] - 0.32, y(q) - 0.3), 0.64, 0.6, fc='white', ec='black', lw=0.6, zorder=2))
            ax.text(x[i], y(q), r'$R_Y$', ha='center', va='center', fontsize=TAMANHO - 2, zorder=3)
    else:
        for a, b in pares(k):
            ax.plot([x[i]] * 2, [y(a), y(b)], color='black', lw=0.8, zorder=2)
            ax.plot([x[i]] * 2, [y(a), y(b)], 'o', color='black', ms=2.5, zorder=3)
xm = x[-1] + 0.8                                   # medição
ax.add_patch(plt.Rectangle((xm - 0.3, y(QUBIT_MEDIDO) - 0.3), 0.6, 0.6, fc='white', ec='black', lw=0.6, zorder=2))
ax.text(xm, y(QUBIT_MEDIDO), r'$Z$', ha='center', va='center', zorder=3)
for l in range(N_CAMADAS):                         # rótulo das camadas
    i0 = 3 * l
    ax.annotate('', xy=(x[i0] - 0.4, N_QUBITS - 0.35), xytext=(x[i0 + 2] + 0.4, N_QUBITS - 0.35),
                arrowprops=dict(arrowstyle='-', lw=0.6))
    ax.text((x[i0] + x[i0 + 2]) / 2, N_QUBITS - 0.15, f'Layer {l + 1}', ha='center', va='bottom')
ax.set_xlim(x[0] - 1.6, xm + 0.6); ax.set_ylim(-0.7, N_QUBITS + 0.3)
ax.axis('off')
salvar(fig, 'fig1_cone')
plt.show()

# %% [md]
# ## 5. Fig. 2: efeito da profundidade (breast cancer)
# Diferença de acurácia da heurística em relação a cada ordenação (pontos percentuais; positivo = heurística melhor, mesmo sinal da Fig. 4 e da Tabela II), com custo local e global. IC de 95% por bootstrap sobre as 10 seeds.
# Escolha em `ORDENS_FIG2` quais ordenações entram.

# %%
ORDENS_FIG2 = ['default', 'reversed', 'random', 'anti_heuristic']
ALTURA = 1.8

fig, axes = plt.subplots(1, 2, figsize=(COLUNA, ALTURA), sharey=True)
for ax, obs, titulo in zip(axes, ['local', 'global'], ['Local cost', 'Global cost']):
    d = depth[depth.m_observable == obs]
    Ls = sorted(d.m_n_layers.unique())
    for o in ORDENS_FIG2:
        r = [diff_ic(d[(d.ordering == 'heuristic') & (d.m_n_layers == L)].test_acc,
                     d[(d.ordering == o) & (d.m_n_layers == L)].test_acc) for L in Ls]
        m, lo, hi = (100 * np.array(v) for v in zip(*r))
        ax.plot(Ls, m, '-o', color=CORES[o], label=NOMES[o])
        ax.fill_between(Ls, lo, hi, color=CORES[o], alpha=0.15, lw=0)
    ax.axhline(0, color='black', lw=0.5)
    ax.set_title(titulo, fontsize=TAMANHO)
    ax.set_xlabel('Layers $L$')
    ax.set_xticks(Ls)
axes[0].set_ylabel('Heuristic − ordering (pp)')
axes[1].legend(loc='upper right', handlelength=1.5)
fig.tight_layout()
salvar(fig, 'fig2_profundidade')
plt.show()

# %% [md]
# ## 6. Fig. 3: variância do gradiente
# Var[∂⟨O⟩/∂θ₀,₀] com IC de 95% por bootstrap: (a) em função de L com 8 qubits; (b) em função do número de qubits com L = 1. Escala log no eixo y.

# %%
ALTURA = 1.8
COR_OBS = {'local': '#1b6ca8', 'global': '#b23a48'}
NOME_OBS = {'local': 'Local cost', 'global': 'Global cost'}

fig, axes = plt.subplots(1, 2, figsize=(COLUNA, ALTURA))
for obs in ['local', 'global']:
    g = grads[(grads.observable == obs) & (grads.n_qubits == 8)].sort_values('n_layers')
    axes[0].errorbar(g.n_layers, g.var_first, yerr=[g.var_first - g.var_first_ci_low, g.var_first_ci_high - g.var_first],
                     fmt='-o', color=COR_OBS[obs], capsize=1.5, elinewidth=0.6, label=NOME_OBS[obs])
    g = grads[(grads.observable == obs) & (grads.n_layers == 1)].sort_values('n_qubits')
    axes[1].errorbar(g.n_qubits, g.var_first, yerr=[g.var_first - g.var_first_ci_low, g.var_first_ci_high - g.var_first],
                     fmt='-o', color=COR_OBS[obs], capsize=1.5, elinewidth=0.6)
for ax in axes:
    ax.set_yscale('log')
axes[0].set_xlabel('Layers $L$ ($n = 8$)')
axes[1].set_xlabel('Qubits $n$ ($L = 1$)')
axes[1].set_xticks(sorted(grads.n_qubits.unique()))
axes[0].set_ylabel(r'Var$[\partial\langle O\rangle/\partial\theta_{0,0}]$')
axes[0].legend(loc='best')
axes[0].set_title('(a)', fontsize=TAMANHO); axes[1].set_title('(b)', fontsize=TAMANHO)
fig.tight_layout()
salvar(fig, 'fig3_gradiente')
plt.show()

# %% [md]
# ## 7. Fig. 4 (opcional): H1 nos 9 datasets com divisão variável
# Diferença heurística − outra ordenação por dataset (pp; positivo = heurística melhor), com IC de 95%. Escolha a profundidade em `L_FIG4`.

# %%
L_FIG4 = 1
ORDENS_FIG4 = ['default', 'reversed', 'random']
ALTURA = 2.6

nomes_ds = list(resplit)
fig, ax = plt.subplots(figsize=(COLUNA, ALTURA))
desloc = np.linspace(-0.22, 0.22, len(ORDENS_FIG4))
for j, o in enumerate(ORDENS_FIG4):
    for i, ds in enumerate(nomes_ds):
        r = resplit[ds]; r = r[r.m_n_layers == L_FIG4]
        m, lo, hi = (100 * v for v in diff_ic(r[r.ordering == 'heuristic'].test_acc, r[r.ordering == o].test_acc))
        ax.errorbar(m, i + desloc[j], xerr=[[m - lo], [hi - m]], fmt='o', color=CORES[o], ms=2.5,
                    elinewidth=0.6, capsize=0, label=NOMES[o] if i == 0 else None)
ax.axvline(0, color='black', lw=0.5)
ax.set_yticks(range(len(nomes_ds)))
ax.set_yticklabels([NOMES_DS.get(d, d) for d in nomes_ds])
ax.invert_yaxis()
ax.set_xlabel(f'Heuristic − ordering (pp), $L = {L_FIG4}$')
ax.legend(loc='lower right', handlelength=1)
fig.tight_layout()
salvar(fig, f'fig4_h1_L{L_FIG4}')
plt.show()

# %% [md]
# ## 8. Tabela II em texto (H1 com divisão variável)
# Média ± desvio padrão da acurácia de teste (%) em 10 seeds, cada uma com sua própria divisão treino-teste.
# Entre parênteses: heurística − ordenação (positivo = heurística melhor); * = Welch unilateral com correção de Holm, p < 0,05.
# Sai em LaTeX (para Overleaf) e em CSV (para colar no Word/Docs).

# %%
ORDENS_TAB = ['default', 'reversed', 'random']

def holm(p):
    p = np.asarray(p); o = np.argsort(p); out = np.empty(len(p)); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1, (len(p) - r) * p[i])); out[i] = run
    return out

linhas = []
for ds, r in resplit.items():
    for L, g in r.groupby('m_n_layers'):
        h = g[g.ordering == 'heuristic'].test_acc
        ps = [stats.ttest_ind(h, g[g.ordering == o].test_acc, equal_var=False, alternative='greater').pvalue for o in ORDENS_TAB]
        ph = holm(ps)
        lin = {'Dataset': NOMES_DS.get(ds, ds), 'L': L, 'Heuristic': f'{100*h.mean():.1f} ± {100*h.std():.1f}'}
        for o, p in zip(ORDENS_TAB, ph):
            v = g[g.ordering == o].test_acc
            lin[NOMES[o]] = f'{100*v.mean():.1f} ({100*(h.mean()-v.mean()):+.1f}){"*" if p < 0.05 else ""}'
        linhas.append(lin)
tab = pd.DataFrame(linhas)
tab.to_csv(f'{OUT}/tabela2.csv', index=False)
with open(f'{OUT}/tabela2.tex', 'w') as f:
    f.write(tab.to_latex(index=False, escape=True, column_format='llcccc'))
print('salvo:', f'{OUT}/tabela2.csv e tabela2.tex')
tab

# %% [md]
# ## 8b. Tabela III em texto (profundidade necessária por tolerância)
# Menor L em que cada ordenação atinge a média em L = 7 (sem a anti-heurística) menos a tolerância. Breast cancer, custo local.

# %%
TOLS = [0.005, 0.01, 0.015, 0.02, 0.03]
loc = depth[depth.m_observable == 'local']
medias = loc.groupby(['ordering', 'm_n_layers']).test_acc.mean().unstack()
alvo = medias[medias.columns.max()].drop('anti_heuristic').mean()
t3 = pd.DataFrame({f'{100*t:g} pp': {NOMES[o]: int(r[r >= alvo - t].index.min()) for o, r in medias.iterrows()} for t in TOLS})
t3 = t3.loc[[NOMES[o] for o in ['heuristic', 'random', 'default', 'reversed', 'anti_heuristic']]]
print(f'alvo = {100*alvo:.1f}% menos a tolerância')
t3.to_csv(f'{OUT}/tabela3.csv')
with open(f'{OUT}/tabela3.tex', 'w') as f:
    f.write(t3.to_latex())
t3

# %% [md]
# ## 9. Baixar tudo

# %%
!cd "{OUT}" && zip -q -r /content/figuras.zip .
from google.colab import files
files.download('/content/figuras.zip')

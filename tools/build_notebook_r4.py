"""Build notebooks/qorder4_experimentos.ipynb (round 4): a self-contained Colab notebook with the whole package embedded.

    python tools/build_notebook_r4.py [out.ipynb]
"""
import base64
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
files = sorted(
    str(p) for p in Path('.').rglob('*')
    if p.is_file() and not any(part in ('results', '__pycache__', 'notebooks', 'tools') for part in p.parts)
    and (p.suffix in ('.py', '.yaml', '.txt') or p.name == 'NOTICE')
)
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
def code(s, form=False):
    cells.append({"cell_type": "code", "metadata": {"cellView": "form"} if form else {}, "execution_count": None,
                  "outputs": [], "source": s.strip("\n").splitlines(True)})

md("""
# qorder · rodada 4

Rode as células **em ordem** (ou "Executar tudo"). Depois de uma desconexão, rode de novo as seções **1** e **3** e continue: o que já foi treinado fica no Drive e é pulado.

| Seção | O que faz | Tempo aproximado |
|---|---|---|
| 1–3 | Código, testes e Drive | ~5 min |
| A | Treino com gradiente estimado por shots (breast cancer), custo local e, como controle, global, **em segundo plano** | ~1,5–2 h, em paralelo com B |
| B | Permutação dentro do cone nos 9 datasets (só as ordens novas) | ~1,5 h |
| C | Espera A terminar e mostra o resumo | — |
| D | Resumo e zip para mandar ao Claude | ~2 min |
| E (opcional) | Avaliação no hardware IBM real (gasta ~2 min da cota) | depende da fila |

O Colab sem GPU serve: estes circuitos são pequenos e a GPU não acelera.
""")
md("## 1. Código e instalação")
lines = ["#@title Criar os arquivos do projeto (clique em ▶)", "import os, base64", "FILES = {"]
for f in files:
    lines.append(f"    {f!r}: {base64.b64encode(open(f, 'rb').read()).decode()!r},")
lines += ["}", "root = '/content/qorder'", "for path, data in FILES.items():",
          "    full = os.path.join(root, path)", "    os.makedirs(os.path.dirname(full), exist_ok=True)",
          "    with open(full, 'wb') as fh:", "        fh.write(base64.b64decode(data))",
          "os.makedirs(os.path.join(root, 'results'), exist_ok=True)",
          "print(f'{len(FILES)} arquivos criados em {root}')"]
code("\n".join(lines), form=True)
code("""
%cd /content/qorder
!pip install -q -r requirements.txt
import pennylane, jax, qiskit
print('PennyLane', pennylane.__version__, '| JAX', jax.__version__, '| Qiskit', qiskit.__version__)
""")
md("## 2. Testes\nResultado esperado: todos `passed`.")
code("!python -m pytest -q")
md("## 3. Pasta de resultados no Google Drive")
code("""
import os
from google.colab import drive
drive.mount('/content/drive')
RESULTS = '/content/drive/MyDrive/qorder_results'
os.makedirs(RESULTS, exist_ok=True)
print('Resultados em', RESULTS)
""")
md("## A. Gradiente por shots (em segundo plano)\nRoda num processo separado, enquanto a seção B treina. O log fica em `qorder_results/exp9_log.txt`.")
code("""
import subprocess
LOG9 = f'{RESULTS}/exp9_log.txt'
proc9 = subprocess.Popen(f'(python -m qorder.run configs/exp9_shots_breast_cancer.yaml --results {RESULTS}; '
                         f'python -m qorder.run configs/exp9b_shots_global.yaml --results {RESULTS}) > {LOG9} 2>&1',
                         shell=True, cwd='/content/qorder')
print('exp9 iniciado em segundo plano; acompanhe com: !tail -3', LOG9)
""")
md("## B. Permutação dentro do cone (9 datasets)\nTreina só `cone_reversed` e `cone_random`; o resto já está no Drive.")
code("""
import glob
for cfg in sorted(glob.glob('configs/exp8_*.yaml')):
    print('=' * 20, cfg, flush=True)
    get_ipython().system(f'python -m qorder.run {cfg} --results {RESULTS}')
""")
md("## C. Esperar a seção A terminar")
code("""
import time
try:
    while proc9.poll() is None:
        print(open(LOG9).read().strip().splitlines()[-1][:120] if os.path.exists(LOG9) else '...', flush=True)
        time.sleep(120)
except NameError:
    print('A seção A não está rodando nesta sessão; rode-a de novo (ela continua de onde parou).')
!tail -3 "$LOG9"
""")
md("## D. Resumo e zip\nGera `qorder_results/rodada4_resumo.txt` e baixa o zip dos resultados (sem os modelos). Mande o zip para o Claude.")
code("""
import glob, os
os.chdir('/content/qorder')
resplit = sorted(d for d in glob.glob(f'{RESULTS}/exp8_*') if os.path.exists(f'{d}/runs.csv'))
OUT = f'{RESULTS}/rodada4_resumo.txt'
with open(OUT, 'w') as f:
    f.write('### Dentro do cone (Wilcoxon por profundidade)\\n')
get_ipython().system(f'python -m qorder.analysis {" ".join(resplit)} --across >> "{OUT}" 2>&1')
get_ipython().system(f'echo "### Shots" >> "{OUT}"')
get_ipython().system(f'python -m qorder.analysis {RESULTS}/exp9_shots_breast_cancer >> "{OUT}" 2>&1')
get_ipython().system(f'python -m qorder.analysis {RESULTS}/exp9b_shots_global >> "{OUT}" 2>&1')
get_ipython().system(f'head -c 4000 "{OUT}"')
get_ipython().system('cd /content/drive/MyDrive && zip -qr qorder_results.zip qorder_results -x "*/models/*"')
from google.colab import files
files.download('/content/drive/MyDrive/qorder_results.zip')
""")
md("""
## E (opcional). Hardware IBM real
Avalia no processador IBM os modelos já treinados de breast cancer (L = 2, seed 0, heurística e padrão), com 1000 shots e 171 amostras. Gasta ~2 min da cota mensal de 10 min. A primeira célula só estima, sem enviar nada.
1. Ícone de **chave** → `IBM_QUANTUM_TOKEN` com **Acesso ao notebook** ativado.
2. Rode a estimativa. 3. Se estiver ok, rode a última célula. A fila da IBM pode levar de minutos a horas.
""")
code("""
from google.colab import userdata
os.environ['IBM_QUANTUM_TOKEN'] = userdata.get('IBM_QUANTUM_TOKEN')
EXP = f'{RESULTS}/exp2_breast_cancer_shallow'
ARGS = '--orderings heuristic default --layers 2 --seeds 0 --shots 1000 --max-samples 200'
!python scripts/validate_hardware.py $EXP --mode ibm:least_busy $ARGS --dry-run
""")
code("""
!python scripts/validate_hardware.py $EXP --mode ibm:least_busy $ARGS --budget 150
!python scripts/summarize_hardware.py $EXP
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}, "colab": {"provenance": [], "toc_visible": True}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'notebooks' / 'qorder4_experimentos.ipynb'
out.parent.mkdir(parents=True, exist_ok=True)
json.dump(nb, open(out, 'w'), indent=1, ensure_ascii=False)
print(len(files), 'files embedded ->', out, f'({out.stat().st_size // 1024} KB)')

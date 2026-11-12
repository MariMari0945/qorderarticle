"""Build notebooks/qorder3_experimentos.ipynb (round 3): a self-contained Colab notebook with the whole package embedded.

    python tools/build_notebook_r3.py [out.ipynb]
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
# qorder · rodada 3 (feedback da Entrega 2)

Rode as células **em ordem**. Depois de uma desconexão, rode de novo as seções **1** e **3** e continue: os modelos já treinados ficam no Drive e são pulados.

| Seção | O que faz | Tempo aproximado |
|---|---|---|
| 1–3 | Código, testes e Drive | ~5 min |
| A | Profundidades L = 4 e L = 6 no breast cancer + sensibilidade da profundidade necessária à tolerância | ~30 min |
| B | Validação com ruído (FakeFez) e shots nas 10 seeds, não só na seed 0 | ~20–40 min |
| C | H1 com divisão treino-teste diferente em cada seed, nos 9 datasets | ~3 h |
| D | Análise e arquivo de resumo para mandar ao Claude | ~2 min |
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
md("## A. Profundidades L = 4 e L = 6 (breast cancer)\nSó treina L = 4 e L = 6: os outros já estão no Drive.")
code("""
!python -m qorder.run configs/exp3_depth_cost.yaml --results $RESULTS
!python -m qorder.analysis $RESULTS/exp3_depth_cost --depth $RESULTS/exp4_gradients
""")
md("## B. Ruído e shots nas 10 seeds\nAvalia os modelos já treinados (breast cancer, L = 2, heurística e padrão). Não treina nada e não usa a cota da IBM.")
code("""
EXP = f'{RESULTS}/exp2_breast_cancer_shallow'
SEEDS = '0 1 2 3 4 5 6 7 8 9'
!python scripts/validate_hardware.py $EXP --mode ideal     --orderings heuristic default --layers 2 --seeds $SEEDS --shots 1000
!python scripts/validate_hardware.py $EXP --mode fake:auto --orderings heuristic default --layers 2 --seeds $SEEDS --shots 1000
!python scripts/summarize_hardware.py $EXP
""")
md("## C. H1 com divisão treino-teste variando com a seed\nNove datasets com ordem padrão arbitrária; cada seed sorteia a própria divisão. Se um dataset falhar, os outros continuam.")
code("""
import glob
for cfg in sorted(glob.glob('configs/exp8_*.yaml')):
    print('=' * 20, cfg, flush=True)
    get_ipython().system(f'python -m qorder.run {cfg} --results {RESULTS}')
""")
md("## D. Análise e resumo\nGera `rodada3_resumo.txt` no Drive (pasta `qorder_results`). Mande esse arquivo para o Claude.")
code("""
import glob, os
resplit = sorted(d for d in glob.glob(f'{RESULTS}/exp8_*') if os.path.exists(f'{d}/runs.csv'))
print('Datasets com divisão variável:', len(resplit))
OUT = f'{RESULTS}/rodada3_resumo.txt'
!echo "### H1 com divisão variável (Wilcoxon por profundidade)" > "$OUT"
!python -m qorder.analysis {' '.join(resplit)} --across >> "$OUT" 2>&1
for d in resplit:
    get_ipython().system(f'echo "### {os.path.basename(d)}" >> "{OUT}"')
    get_ipython().system(f'python -m qorder.analysis "{d}" >> "{OUT}" 2>&1')
!echo "### Profundidade necessária" >> "$OUT"
!python -m qorder.analysis $RESULTS/exp3_depth_cost --depth $RESULTS/exp4_gradients >> "$OUT" 2>&1
!echo "### Ruído e shots" >> "$OUT"
!python scripts/summarize_hardware.py $RESULTS/exp2_breast_cancer_shallow >> "$OUT" 2>&1
!head -c 6000 "$OUT"
""")
code("""
# Opcional: baixar todos os resultados (sem os modelos salvos), para o Claude recalcular tudo
!cd /content/drive/MyDrive && zip -qr qorder_results.zip qorder_results -x "*/models/*"
from google.colab import files
files.download('/content/drive/MyDrive/qorder_results.zip')
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}, "colab": {"provenance": [], "toc_visible": True}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'notebooks' / 'qorder3_experimentos.ipynb'
out.parent.mkdir(parents=True, exist_ok=True)
json.dump(nb, open(out, 'w'), indent=1, ensure_ascii=False)
print(len(files), 'files embedded ->', out, f'({out.stat().st_size // 1024} KB)')

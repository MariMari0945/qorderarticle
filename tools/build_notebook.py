"""Build notebooks/qorder_experimentos.ipynb: a self-contained Colab notebook with the whole package embedded.

    python tools/build_notebook.py [out.ipynb]
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
# qorder: experimentos do artigo

Rode as células **em ordem**. Depois de uma desconexão, rode de novo as seções **1** e **3** e continue de onde parou: os modelos já treinados ficam no Drive e são pulados.

- **1.** Cria o código e instala as bibliotecas (~3 min).
- **2.** Testes. Se algum falhar, pare e mande a saída.
- **3.** Liga o Google Drive (clique em "Conectar" e marque todas as permissões).
- **4–5.** Experimentos da primeira rodada (já rodados: serão pulados).
- **5b.** Novos testes da H1.
- **6.** Análise. **7.** Validação com ruído e hardware.
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
md("## 4. Teste rápido do pipeline")
code("!python -m qorder.run configs/exp2_breast_cancer_shallow.yaml --quick --results $RESULTS")
md("## 5. Experimentos da primeira rodada\nJá rodados: estas células só treinam o que faltar.")
code("!python -m qorder.run configs/exp2_breast_cancer_shallow.yaml --results $RESULTS")
code("!python -m qorder.run configs/exp4_gradients.yaml --results $RESULTS")
code("!python -m qorder.run configs/exp3_depth_cost.yaml --results $RESULTS")
code("!python -m qorder.run configs/exp2_mnist_shallow.yaml --results $RESULTS\n!python -m qorder.run configs/exp2_fashion_shallow.yaml --results $RESULTS")
code("!python -m qorder.run configs/exp1_adversarial_drc.yaml --results $RESULTS")
md("""
## 5b. Novos testes da H1 e linhas de base pedidas pelos professores

Cada célula só treina o que ainda não existe no Drive. Rode na ordem.

| Célula | O que testa | Tempo aproximado |
|---|---|---|
| A | 6 datasets tabulares com ordem arbitrária: todas as ordens + linhas de base (k melhores, qubit medido no centro) + mRMR | ~4 h |
| B | MNIST e Fashion com componentes da PCA embaralhados, com as mesmas linhas de base | ~1 h 30 |
| C | Breast cancer e experimentos anteriores: completa com correlação, mRMR e linhas de base | ~1 h |
| D | Dataset adversarial com 1000 amostras de treino | ~1 h 30 |\n| E | Variância do gradiente de novo, agora com intervalo de confiança (bootstrap) para a H3 | ~20 min |

**Linhas de base novas** (feedback da Sprint 1):
- `kbest`: circuito menor, só com as L+1 features mais informativas (todas dentro do cone).
- `default_center` / `heuristic_center`: qubit medido no centro da cadeia (q3), o que dobra o alcance do cone.
- `heuristic_mrmr`: heurística com mRMR, que desconta redundância entre features.
""")
code("""
# A · 6 datasets tabulares (se um dataset falhar ao baixar, os outros continuam)
import glob, subprocess
for cfg in sorted(glob.glob('configs/exp5_*.yaml')):
    print('=' * 20, cfg, flush=True)
    r = subprocess.run(['python', '-m', 'qorder.run', cfg, '--results', RESULTS])
    if r.returncode != 0:
        print('>>> falhou:', cfg, '(continuando)')
""")
code("# B · MNIST e Fashion com componentes embaralhados\n!python -m qorder.run configs/exp6_mnist_shuffled.yaml --results $RESULTS\n!python -m qorder.run configs/exp6_fashion_shuffled.yaml --results $RESULTS")
code("# C · completa os experimentos anteriores (só os modelos novos)\n!python -m qorder.run configs/exp2_breast_cancer_shallow.yaml --results $RESULTS\n!python -m qorder.run configs/exp2_mnist_shallow.yaml --results $RESULTS\n!python -m qorder.run configs/exp2_fashion_shallow.yaml --results $RESULTS\n!python -m qorder.run configs/exp1_adversarial_drc.yaml --results $RESULTS")
code("# D · adversarial com 1000 amostras de treino\n!python -m qorder.run configs/exp7_adversarial_n1000.yaml --results $RESULTS")
code("# E · variância do gradiente com intervalo de confiança (bootstrap), para a H3\n!python -m qorder.run configs/exp4_gradients.yaml --results $RESULTS")
md("## 6. Análise\nPrimeiro a H1 em todos os datasets com ordem padrão arbitrária; depois cada experimento.")
code("""
import glob
arb = [f'{RESULTS}/exp2_breast_cancer_shallow'] + sorted(glob.glob(f'{RESULTS}/exp5_*')) + sorted(glob.glob(f'{RESULTS}/exp6_*'))
arb = [d for d in arb if os.path.exists(f'{d}/runs.csv')]
print('H1 · datasets com ordem padrão arbitrária:', len(arb))
!python -m qorder.analysis {' '.join(arb)} --across
""")
code("""
# H3 reformulada: profundidade necessária para cada ordem atingir a acurácia do regime de cone completo,
# e a variância do gradiente nessa profundidade
!python -m qorder.analysis $RESULTS/exp3_depth_cost --depth $RESULTS/exp4_gradients
""")
code("""
from IPython.display import Image, display
for d in sorted(glob.glob(f'{RESULTS}/exp*')):
    if d.endswith('_quick'):
        continue
    print('=' * 25, os.path.basename(d))
    !python -m qorder.analysis "$d"
    for img in ('orderings.png', 'gradients.png'):
        if os.path.exists(f'{d}/{img}'):
            display(Image(f'{d}/{img}'))
""")
md("## 7. Validação: shots e ruído simulado (gratuito)")
code("""
EXP = f'{RESULTS}/exp2_breast_cancer_shallow'
!python scripts/validate_hardware.py $EXP --mode ideal     --orderings heuristic default --layers 2 --shots 1000
!python scripts/validate_hardware.py $EXP --mode fake:auto --orderings heuristic default --layers 2 --shots 1000
""")
md("""
### Hardware IBM (gasta a cota de 10 min/mês)
1. Ícone de **chave** → adicione `IBM_QUANTUM_TOKEN` e ative **Acesso ao notebook**.
2. Rode a célula de estimativa **manualmente** (ela não envia nada).
3. Só depois de o grupo decidir, tire o `#` da última célula.
""")
code("""
from google.colab import userdata
os.environ['IBM_QUANTUM_TOKEN'] = userdata.get('IBM_QUANTUM_TOKEN')
ARGS = '--orderings heuristic default --layers 2 --seeds 0 --shots 1000 --max-samples 200'
!python scripts/validate_hardware.py $EXP --mode ibm:least_busy $ARGS --dry-run
""")
code("# Envia de verdade: tire o # quando o grupo decidir\n# !python scripts/validate_hardware.py $EXP --mode ibm:least_busy $ARGS --budget 150")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}, "colab": {"provenance": [], "toc_visible": True}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'notebooks' / 'qorder_experimentos.ipynb'
out.parent.mkdir(parents=True, exist_ok=True)
json.dump(nb, open(out, 'w'), indent=1, ensure_ascii=False)
print(len(files), 'files embedded ->', out, f'({out.stat().st_size // 1024} KB)')

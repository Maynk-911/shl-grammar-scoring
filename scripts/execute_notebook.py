"""Execute all notebook cells using this interpreter, keeping runtime files local."""
from pathlib import Path
import os
import sys
import json
import asyncio
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
runtime = root / '.jupyter'
runtime.mkdir(exist_ok=True)
kernel = runtime / 'kernels' / 'shl-local'
kernel.mkdir(parents=True, exist_ok=True)
(kernel/'kernel.json').write_text(json.dumps({
    'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    'display_name': 'SHL local', 'language': 'python'}))
os.environ['JUPYTER_PATH'] = str(runtime)
os.environ['JUPYTER_RUNTIME_DIR'] = str(runtime / 'runtime')
os.environ['IPYTHONDIR'] = str(runtime / 'ipython')
path = root / 'SHL_Grammar_Scoring.ipynb'
notebook = nbformat.read(path, as_version=4)
client = NotebookClient(notebook, timeout=1800, kernel_name='shl-local',
                        resources={'metadata': {'path': str(root)}})
def cell_done(cell, cell_index, **kwargs):
    if cell.cell_type == 'code':
        print(f'Executed code cell {cell_index+1}/{len(notebook.cells)}', flush=True)
client.on_cell_executed = cell_done
try:
    client.execute()
finally:
    nbformat.write(notebook, path)
errors = [o for c in notebook.cells if c.cell_type == 'code' for o in c.get('outputs', []) if o.output_type == 'error']
assert not errors, errors
print('Notebook executed top to bottom without errors.', flush=True)

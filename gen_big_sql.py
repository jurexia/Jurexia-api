"""Output SQL INSERT statements for all juzgados, 100 rows per block, to stdout."""
import json, sys

with open('juzgados_distrito.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

def esc(s):
    if not s:
        return 'NULL'
    return "'" + s.replace("'", "''") + "'"

# Output batches of 100 to separate files
batch_size = 100
for batch_idx in range(0, len(data), batch_size):
    batch = data[batch_idx:batch_idx+batch_size]
    filename = f'big_batch_{batch_idx//batch_size}.sql'
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('INSERT INTO juzgados_distrito (denominacion,materia,circuito,estado,ciudad,direccion,telefono) VALUES\n')
        rows = []
        for d in batch:
            ciudad = d.get('estado', '')
            rows.append(f"({esc(d['denominacion'])},{esc(d['materia'])},{d['circuito']},{esc(d['estado'])},{esc(ciudad)},{esc(d['direccion'])},{esc(d['telefono'])})")
        f.write(',\n'.join(rows) + ';\n')
    print(f'{filename}: {len(batch)} rows, {len(open(filename, encoding="utf-8").read())} chars')

import json

with open('juzgados_distrito.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Build clean SQL with proper escaping
def esc(s):
    if not s:
        return 'NULL'
    return "'" + s.replace("'", "''").replace('"', '"') + "'"

batch_size = 25  # Smaller batches for MCP
for batch_num in range(0, len(data), batch_size):
    batch = data[batch_num:batch_num+batch_size]
    filename = f'mini_batch_{batch_num//batch_size}.sql'
    lines = ['INSERT INTO juzgados_distrito (denominacion,materia,circuito,estado,ciudad,direccion,telefono) VALUES']
    rows = []
    for d in batch:
        # Fix ciudad - use estado as fallback
        ciudad = d.get('estado', '')
        row = f"({esc(d['denominacion'])},{esc(d['materia'])},{d['circuito']},{esc(d['estado'])},{esc(ciudad)},{esc(d['direccion'])},{esc(d['telefono'])})"
        rows.append(row)
    lines.append(',\n'.join(rows) + ';')
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'{filename}: {len(batch)} rows')

total_batches = (len(data)-1)//batch_size + 1
print(f'Total: {len(data)} rows in {total_batches} batches')

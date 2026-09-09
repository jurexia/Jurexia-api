"""LO QUE LOS AUTOS YA DICEN, PARA NO PREGUNTARLO.

David lo señaló al describir el material de cada asunto: «también está el auto
de admisión y el auto de turno que son indispensables para verificar datos como
la presentación, los terceros interesados, el magistrado ponente». Tenía razón,
y son más datos de los que parecía. Medido sobre los dos autos reales del
91/2025:

  auto de admisión (21/11/2025)
      número de expediente ..... 91/2025
      recurrente ............... Karen Yadira Meza Ruiz, Administradora
                                 Desconcentrada Jurídica de Querétaro «1», SAT
      expediente de origen ..... 695/25-09-01-7-OT
      secretario de acuerdos ... Omar Alejandro Elizalde Herrera
  auto de turno (16/01/2026)
      magistrado ponente ....... Luis Armando Perez Topete

Todo eso se teclea hoy. Nada de eso hace falta teclear.

LAS FECHAS VAN EN LETRA, que es como se escriben los autos: «veintiuno de
noviembre de dos mil veinticinco». Por eso hay aquí un lector de números en
palabras y no una expresión regular de dígitos.

Y UNA RAYA QUE NO SE CRUZA. Esto LEE; no decide. Lo que salga de aquí llega al
secretario como propuesta con su origen a la vista, para que lo confirme. La
fecha de notificación —la que decide si el recurso es extemporáneo— es
justamente la que estos autos NO traen, y suponerla es el fallo que dejó a
Erika con dos proyectos vacíos por extemporaneidad. Cuando no se lee, se dice.
"""

import re
import unicodedata

UNIDADES = {
    "cero": 0, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
    "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10, "once": 11,
    "doce": 12, "trece": 13, "catorce": 14, "quince": 15, "dieciseis": 16,
    "diecisiete": 17, "dieciocho": 18, "diecinueve": 19, "veinte": 20,
    "veintiuno": 21, "veintidos": 22, "veintitres": 23, "veinticuatro": 24,
    "veinticinco": 25, "veintiseis": 26, "veintisiete": 27, "veintiocho": 28,
    "veintinueve": 29, "treinta": 30, "treintaiuno": 31,
}
MESES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}


def _sin_tildes(t: str) -> str:
    t = unicodedata.normalize("NFKD", t)
    return "".join(c for c in t if not unicodedata.combining(c))


def _dia(palabras: str):
    p = _sin_tildes(palabras.lower()).strip()
    if p in UNIDADES:
        return UNIDADES[p]
    # «treinta y uno», «veinte y dos»
    m = re.fullmatch(r"(treinta|veinte)\s+y\s+(\w+)", p)
    if m and m.group(2) in UNIDADES:
        return UNIDADES[m.group(1)] + UNIDADES[m.group(2)]
    return None


def _anio(palabras: str):
    p = _sin_tildes(palabras.lower())
    if not p.startswith("dos mil"):
        return None
    resto = p[len("dos mil"):].strip()
    if not resto:
        return 2000
    # «veinticinco», «veinte», «veinte y cinco»
    n = _dia(resto)
    return 2000 + n if n is not None else None


_RX_FECHA = re.compile(
    r"\b((?:treinta|veinte)\s+y\s+\w+|\w+)\s+de\s+(" + "|".join(MESES) + r")"
    r"\s+de\s+(dos\s+mil(?:\s+(?:\w+\s+y\s+\w+|\w+))?)", re.I)


def fechas(texto: str, limite: int = 6):
    """Las fechas escritas en letra, en el orden en que aparecen.

    Devuelve [(iso, literal, posición)]. El orden importa: la primera fecha de
    un auto es casi siempre la suya.
    """
    salida = []
    for m in _RX_FECHA.finditer(_sin_tildes(texto)):
        d, mes, a = _dia(m.group(1)), MESES.get(m.group(2).lower()), _anio(m.group(3))
        if d and mes and a and 1 <= d <= 31:
            salida.append((f"{a:04d}-{mes:02d}-{d:02d}",
                           m.group(0).strip(), m.start()))
        if len(salida) >= limite:
            break
    return salida


def _limpiar_nombre(n: str) -> str:
    n = re.sub(r"\s+", " ", n or "").strip(" ,.;:")
    # El OCR mete el escudo y el sello donde puede.
    n = re.sub(r"\b(PODER JUDICIAL|DE LA FEDERACION|FORMA B|UNIDOS|ESTADOS)\b.*",
               "", n, flags=re.I).strip(" ,.;:")
    return n if 6 <= len(n) <= 90 else ""


def leer(texto: str) -> dict:
    """Lo que se puede leer de un auto. Lo que no, no sale."""
    d = {}
    fs = fechas(texto)
    if fs:
        d["fecha_auto"] = fs[0][0]
        d["fecha_auto_literal"] = fs[0][1]

    m = re.search(r"f[óo]rmese el expediente n[úu]mero\s+([\d]{1,5}/20\d{2})", texto, re.I)
    if m:
        d["numero"] = m.group(1)

    # 695/25-09-01-7-OT · el juicio del que viene el recurso
    m = re.search(r"\b(\d{3,5}/\d{2}-\d{2}-\d{2}-\d(?:-[A-Z]{2})?)\b", texto)
    if m:
        d["expediente_origen"] = m.group(1)

    # EL PONENTE. «TURNESE … a la ponencia A MI CARGO» y quien firma abajo es
    # el Presidente: ése es el ponente. La otra forma —«a la ponencia del
    # magistrado Fulano»— también se contempla.
    m = re.search(r"ponencia\s+(?:del|de la)\s+(?:magistrad[oa]\s+)?([A-ZÁÉÍÓÚÑ][\w áéíóúñÁÉÍÓÚÑ.]{6,70})",
                  texto)
    if m:
        d["magistrado"] = _limpiar_nombre(m.group(1))
    elif re.search(r"ponencia\s+a\s+mi\s+cargo", texto, re.I):
        f = re.search(r"firma\s+([A-ZÁÉÍÓÚÑ][\w áéíóúñÁÉÍÓÚÑ.]{6,70}?),?\s+"
                      r"Magistrad[oa]\s+Presidente", texto, re.I)
        if f:
            d["magistrado"] = _limpiar_nombre(f.group(1))
            d["magistrado_de"] = "ponencia a mi cargo · firma el Presidente"

    # EL NOMBRE COMPLETO, NO EL FINAL. Con una captura perezosa salía
    # «Alejandro Elizalde Herrera» y se dejaba el «Omar»: se toman las palabras
    # capitalizadas contiguas que preceden a la mención del cargo.
    m = re.search(r"((?:[A-ZÁÉÍÓÚÑ][a-záéíóúñ.]+\s+){1,5}[A-ZÁÉÍÓÚÑ][a-záéíóúñ.]+)"
                  r"\s*,\s*Secretari[oa]\s+de\s+Acuerdos", texto)
    if m:
        d["secretario"] = _limpiar_nombre(m.group(1))

    # EL RECURRENTE ES LA PERSONA, no su cargo entero. En el 91/2025 el cargo
    # sigue durante ciento veinte caracteres —«Administradora Desconcentrada
    # Jurídica de Querétaro «1», con sede en…»— y descartaba el nombre por
    # largo. Se corta en la primera coma y el cargo se guarda aparte.
    m = re.search(r"agravios\s+presentad[oa]\s+por\s+([^;.]{8,200})", texto, re.I)
    if m:
        entero = re.sub(r"\s+", " ", m.group(1)).strip()
        d["recurrente"] = _limpiar_nombre(entero.split(",")[0])
        if "," in entero:
            d["recurrente_cargo"] = entero.split(",", 1)[1].strip()[:160]

    # LA PORTADA DE LA OFICINA DE CORRESPONDENCIA COMÚN trae la fecha de
    # presentación en dígitos, y no es la misma que la de ingreso al tribunal.
    # Medido en el 91/2025: presentación 13/11/2025, ingreso 18/11/2025. Cinco
    # días. Computar desde el ingreso corre el plazo hacia adelante y puede
    # volver extemporáneo un recurso que no lo era: el mismo daño que dejó dos
    # proyectos vacíos, por otro camino.
    m = re.search(r"Fecha\s+de\s+presentaci[óo]n[^:]{0,30}:\s*(\d{1,2}/\d{1,2}/20\d{2})",
                  texto, re.I)
    if m:
        d_, m_, a_ = m.group(1).split("/")
        d["presentacion"] = f"{a_}-{int(m_):02d}-{int(d_):02d}"
        d["presentacion_de"] = "portada de la Oficina de Correspondencia Común"
    m = re.search(r"Folio\s+electr[óo]nico:\s*(\d{4,12})", texto, re.I)
    if m:
        d["folio"] = m.group(1)

    if re.search(r"\bT[UÚ]RNESE\b", texto, re.I):
        d["es_turno"] = True
    if re.search(r"reg[íi]strese y f[óo]rmese el expediente", texto, re.I):
        d["es_admision"] = True
    if re.search(r"citaci[óo]n para sentencia", texto, re.I):
        d["cita_para_sentencia"] = True
    return d


def juntar(autos):
    """Un solo cuadro con lo leído en varios autos. Gana el que trae el dato;
    si dos lo traen distinto, se conserva el primero y se anota la discrepancia,
    porque una contradicción entre constancias es cosa del secretario."""
    fin, choques = {}, []
    for a in autos:
        for k, v in (a or {}).items():
            if not v:
                continue
            if k not in fin:
                fin[k] = v
            elif fin[k] != v and k in ("numero", "magistrado", "expediente_origen"):
                choques.append(f"{k}: «{fin[k]}» y «{v}»")
    if choques:
        fin["discrepancias"] = choques
    return fin

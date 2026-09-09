# DÓNDE VAMOS · 9-sep, 05:50

Fichero de rescate: si la sesión se corta otra vez, esto es lo que hay que
saber para seguir sin preguntar nada.

## Hecho y probado

**La extensión v2, contra la API del Expediente Electrónico.** Se tiró el
forcejeo con el ASP.NET de SISE. Contrato medido en `SISE-MAPA.md`. Prueba de
extremo a extremo con el 91/2025: 9 documentos en el índice, 3 marcados por la
regla, 3 traídos (5 MB), HTTP 200 en 27 s, fila leída de vuelta de Supabase.
El clasificador acertó los tres: `sentencia_recurrida`, `auto_admision`,
`auto_turno`.

**Revisión adversarial** (`wf_fba795cb-e57`): 5 lentes, 44 agentes; 13
terminaron y 31 murieron con el límite de sesión. UN hallazgo confirmado y
grave —panel reentrante: cerrar y reabrir a mitad de envío dejaba dos pasadas
vivas y el upsert se quedaba con la mala, sin ningún mensaje—. **Ya aplicado.**
Los 31 títulos sin verificar están en el aviso de la tarea; no se tocan sin
comprobarlos, que es la regla de siempre.

## Lo que falta

1. **DEPURACIÓN para el OCR.** Palabras de David: «Es fundamental que depures
   los documentos para la correcta lectura del pipeline OCR vía Azure. Sin
   esto, la calidad del proyecto no será la misma.»

   PRIMERA MEDICIÓN, ya hecha sobre el escaneo real del 91/2025:

       promocion.pdf ..... 117 páginas, 3.743 KB
       texto nativo ...... 1 página de 117
       caracteres nativos . 12.625 — y son BASURA:

   Las 117 páginas llevan la MISMA cadena de 91 caracteres incrustada:
   `MARIA DE LA LUZ RAMIREZ MARTINEZ 706a6620636a6632…` — el sello de la firma
   electrónica del CJF (ese hex es «pjf cjf2»). No es contenido: es el
   estampado que el sistema pone encima de cada hoja.

   Ahí está el problema que hay que resolver, y sigue sin medirse del todo:
   dónde acaba la sentencia recurrida y dónde empiezan los agravios dentro de
   esas 117 páginas, y qué páginas son puro acuse, sello o cara en blanco.

2. **Del expediente al proyecto, sin formulario.** Que el taller ofrezca
   generar desde lo descargado, con el paso de depuración delante, y que con lo
   poco que falte lleve directo a: contexto → problemas jurídicos → el
   secretario elige el sentido.

3. **Cinco sentencias distintas**, el resultado de cada una, y mi propia
   valoración de la calidad y de qué falta.

## Disciplinas que no se negocian

Medir antes de tocar. Probar el camino completo, no cada pieza. No dar por
bueno lo que no se ha corrido. Una comprobación que acusa al trabajo correcto
es la comprobación que está mal.

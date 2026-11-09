# CLAUDE.md

Antes de trabajar en el repo, leer [CONTEXTO.md](CONTEXTO.md). Tiene el estado del proyecto, el entorno
(Windows, venv con uv) y los archivos clave.

## Prescripciones y archivos de salida para monitores

Al tocar `src/maleza/prescripcion.py`, los formatos de export o la estructura de las salidas para
pulverizadoras, usar [docs/prescripciones_monitores.md](docs/prescripciones_monitores.md) como
referencia. Ahí están el formato y la carpeta de pendrive por monitor, los límites de polígonos, el
comportamiento de la dosis 0 y los requisitos pendientes del exportador.

Reglas:

- No inventar formatos, carpetas ni nombres de atributos de un monitor. Si no figuran en el doc con
  fuente, marcarlo como "sin verificar" y preguntar.
- Ningún export se probó todavía en un monitor real. No presentar un perfil como validado hasta que el
  usuario confirme la prueba en máquina. Cuando lo confirme, anotarlo en el doc.
- Si aparece información nueva (manual, prueba en campo, respuesta de un fabricante), actualizar el doc
  con la fuente y la fecha.

# Prescripciones para monitores de pulverizadoras (Argentina)

Relevamiento del 2026-11-06. Sirve de referencia para `src/maleza/prescripcion.py`: qué archivo y qué
estructura de pendrive espera cada monitor. Las fuentes están al final. Lo que no tiene fuente está
marcado como **sin verificar**. No hay que completarlo de memoria.

## Resumen

- Casi todos los monitores del parque argentino leen **Shapefile** (.shp/.shx/.dbf; el .prj es opcional).
  Lo que cambia es la **carpeta del pendrive** y algunos **límites**.
- Hay excepciones. **ISOXML (TASKDATA)**: AGCO, Massey, Valtra y TeeJet (este último vía Fieldware Link).
  **Conversión propietaria**: ControlAgro (MHS), TIM (proMAP), Verion (Agromapas).
- No hay que entregar un shapefile genérico, sino un **perfil por monitor**. Así lo hacen FieldView y
  Avansys: la geometría base es una sola y cambian la carpeta, el formato y los ajustes.
- El problema de fondo es cómo cada monitor interpreta la **dosis 0** y los **polígonos chicos** para
  cortar secciones, más que el formato del archivo.

## Tabla por monitor

| Monitor | Formato | Estructura en el pendrive | Notas |
| --- | --- | --- | --- |
| Case IH Pro 700 / NH IntelliView III-IV | Shapefile | `Shapefile/` en la raíz, junto a la carpeta .cn1 | Firmware V30–V36 (pestaña Import2 en Gestión de datos). Máx. 10 Rx y **1000 polígonos por lote**. Configurar "Out of Zone Rate". Nombre de carpeta exacto: "Shapefile", sin "s" final. Pro 600 solo .cn1 |
| Case IH Pro 1200 / NH IntelliView 12 | Shapefile | Archivos en la raíz | FieldView exporta "Generic (Shapefile)" |
| John Deere GS 2630 / Gen 4 (4640, 4600) / G5 | Shapefile | `Rx/` en la raíz, con todos los lotes adentro | Dosis 0 apaga secciones con Section Control (documentado por Deere). También entra por Operations Center. Las Pla nuevas traen Gen 4 4640 |
| John Deere GS 2600 | Shapefile | Raíz | — |
| Trimble GFX-350/750/1060/1260, TMX-2050 (Precision IQ) | Shapefile | `AgData/Prescriptions/` | Ver los problemas de on/off más abajo |
| Trimble FmX / CFX-750 | Shapefile | `AgGPS/Prescriptions/` | CFX-750 necesita el desbloqueo de dosis variable. FmX con factor de escala 1 |
| Raven Viper 4/4+ / Viper Pro / Envizio Pro | Shapefile | `Raven/Misc/rxMaps/` (Viper) | Viper Pro: las zonas a 0 se configuran como "no-spray zones" |
| Ag Leader InCommand 800/1200, Integra | Shapefile | Raíz o una carpeta en la raíz | Hay un reporte de que no corta secciones en manchones menores al ancho del botalón (**sin verificar** con el firmware actual) |
| Topcon X25/X30/X35 | Shapefile | `VRC/` | — |
| Stara Topper 4500/5500/6500 | Shapefile | `Dados/Mapas/` | También se carga por Telemetría Stara |
| Jacto Omni 700 | Shapefile | Raíz | — |
| Müller Touch 800/1200 | Shapefile | `SHP/` en la raíz | Requiere activar la licencia |
| Hexagon Ti5/Ti7/Ti10 | Shapefile | Raíz | — |
| ARAG Bravo 400S / Delta 80 | Shapefile | `maps/` en la raíz | — |
| Abelardo Cuffia AG9500 / AG Fusion / Fusion Pro | Shapefile | Según la guía del fabricante | **Máx. 50 polígonos (AG9500) y 100 (AG Fusion)** |
| Plantium SBOX 7/11 | Shapefile | Raíz | Permite elegir la columna de dosis |
| Microelec (fertilización/pulverización) | Shapefile | Sin especificar | — |
| ControlAgro CAS 4500 | .SFV (convertido) | — | Convertir con el software MHS |
| TIM CS Oro Máx / TS Oro | .srm (convertido) | Raíz de la tarjeta / tablet | Convertir con proMAP. Requiere código de activación |
| Verion Vcom 7/10 | Convertido | — | Convertir con Agromapas 4.0 |
| AGCO C2000/C3000, Massey DataTronic 5, Valtra SmartTouch | ISOXML | `TASKDATA/` en la raíz | Un solo TASKDATA con todos los lotes |
| TeeJet Matrix Pro GS / Aeros | ISOXML vía Fieldware Link | Trabajo exportado por Fieldware Link | La Rx se importa en la PC y después se exporta el trabajo a la consola |
| Sensor, Caimán (monitor propio), Metalfor | **sin verificar** | **sin verificar** | Sin documentación pública. Avansys declara exportar a Sensor y Caimán |

## Corte de secciones, dosis 0 y fuera de zona

- **John Deere Gen 4:** "Zero rates in the prescription command sections OFF in Section Control".
- **Raven Viper Pro:** según una guía, las zonas a 0 se tienen que configurar como "no-spray zones".
  Así como vienen en la Rx, no cortan.
- **Case IH Pro 700 / IntelliView IV:** al salir de la prescripción aplica el "Out of Zone Rate". Hay un
  usuario que reporta que el monitor mostraba bien 0 y 35–40 L/ha, pero no cortaba secciones ni hacía
  bien las transiciones. **No hay confirmación oficial** de que una dosis 0 corte secciones en
  pulverizadora.
- **Trimble GFX-750:** hay reportes de secciones que no cortan con dosis 0. Además, una zona de aplicación
  en forma de anillo (con hueco) no encendía las secciones del centro; sin huecos funcionaba. Pix4D
  indica que la Rx on/off se carga como **límite** en `AgGPS\Data\<productor>\<lote>`, con columnas
  específicas (**sin verificar** cuáles).
- **Ag Leader:** ver la tabla.

## Qué hace Avansys (referencia de competencia)

- Monitores declarados: Trimble, John Deere, Case IH, New Holland, Raven, Stara, Jacto, Caimán, Müller y
  Sensor. Drones: DJI Agras, XAG y TOP GUN. Para el resto: SHP, KML y GeoJSON.
- Exporta un archivo por monitor (por ejemplo "Trimble GFX-750.shp" y "John Deere Gen 4.shp").
- La grilla del mapa de malezas es de 5 m en los casos publicados.
- La versión Desktop v1.0.34 agrega un buffer "A medida" en metros y "la dosis fuera de la maleza para
  las marcas que la soportan". Eso confirma que el manejo del fondo depende de la marca.
- No publica carpetas, atributos, proyección ni qué marcas aceptan la dosis fuera de zona.

## Caso real: lote26 (IntelliView IV, 2026-11-05)

El shapefile `lote26` salió de esta app con la casilla de fondo desmarcada:

- Una sola fila multiparte (`zona=maleza`, `rx=100`, `unidad=L/ha`), con 271 partes y 17 huecos.
- 210 partes miden menos de 100 m² (mediana de 38 m²).
- 53.000 vértices con segmentos de unos 25 cm.
- Proyección UTM 20S en lugar de WGS84.
- **Sin zona a 0**, así que el monitor aplica el "Out of Zone Rate" en todo el resto del lote.

Correcciones aplicadas a mano: pasar a WGS84, separar en polígonos simples, simplificar a 0,2 m (el área
bajó menos del 1%), usar el campo `RATE` y armar la carpeta `Shapefile/`.

## Archivo de Avansys que funciona en IntelliView IV como límite (2026-11-08)

Fuente: el usuario confirmó que estos shapefiles, generados por Avansys para "INTELLIVIEWIV", funcionan en una
IntelliView IV **cargados como límite** (no como Rx). Son dos archivos (`20260611-LOTE_1` y `20261108-LOTE_28`), con
sus reportes PDF ("El archivo contiene solo las áreas donde se debe aplicar herbicida"). Analizados byte a byte:

- Carpeta `Shapefile/`. Nombre `AAAAMMDD-LOTE_N` (fecha de generación + nombre del lote).
- `.shp` tipo 5 (Polygon), **un solo registro** multiparte con toda la maleza (93 y 45 partes).
- **Con huecos**: 8 y 13 anillos interiores. Exteriores en sentido horario, huecos antihorario (el estándar).
- **Solo la zona de maleza**: no hay fondo ni zona a 0 (3,24 de 22,97 ha y 5,65 de 25,34 ha).
- `.dbf` dBase III con un solo campo `rx`, N 5.4, con valor `0.000`. La dosis del reporte (80 y 60 L/ha) no
  está en el archivo.
- `.prj` WGS84 en WKT de ESRI (`GCS_WGS_1984`). `.cpg` con `ISO-8859-1`.
- Coordenadas con precisión doble completa. Grilla de 5 m girada según el lote (la grilla de la app es aparte).

El perfil `case_pro700` copia todo esto salvo la grilla (`as_boundary=True`). **Nuestro export no se probó
todavía en la máquina.**

## Requisitos para el exportador (implementados el 2026-11-06, sin probar en máquina)

Están en `PROFILES`, `adapt_to_monitor` y `export_for_monitor` de `prescripcion.py`. Se implementaron con
este doc como fuente, pero **ningún perfil se probó todavía en un monitor real**.

1. **Perfiles por monitor** (`PROFILES`): formato, carpeta del pendrive, máximo de polígonos, si admite
   huecos y si lleva fondo a 0. Están en el código, no en `config.yaml`. Los monitores que requieren
   conversor o sin documentación figuran deshabilitados en el desplegable.
2. **Geometría base:** WGS84 con .prj, con coordenadas redondeadas a 1e-8° (~1 mm) y geometría reparada: al
   reproyectar, un hueco que toca un borde podía quedar autointersectado (corregido el 2026-11-06). Un solo atributo, `RATE`, escrito como N 10.2. Se usa el mismo
   nombre en todos los perfiles porque es el del caso lote26; **sin verificar** en los demás monitores.
3. **Fondo solo si se pide** (cambio del 2026-11-08, a pedido del usuario: antes los perfiles de monitor con
   shapefile agregaban el fondo a 0 aunque la casilla estuviera desmarcada). El perfil de límite de
   Pro 700 / IntelliView nunca lleva fondo. Para Trimble, la nota del perfil indica marcar el fondo a 0.
4. **Polígonos simples:** una fila por polígono en todos los shapefiles. **Pendiente de probar** cuál lee
   mejor cada monitor.
5. **Tamaño mínimo de manchón:** ancho de sección × velocidad × retardo (sliders que se activan con una casilla; por defecto no se aplica).
   Los manchones más chicos se agrandan hasta esa superficie, no se descartan, para no perder maleza.
6. **Simplificar** a 0,2 m y **controlar el conteo:** al superar el máximo del perfil, la app avisa y pide
   confirmación antes de exportar. No agranda el buffer por su cuenta. El conteo incluye las filas de fondo.
7. **Maleza sin anillos** en Trimble (en Pro 700 / IntelliView III-IV ya no: desde el 2026-11-08 copia el archivo
   de Avansys, que lleva huecos y funciona como límite): se parten los anillos de la zona de
   maleza, que es donde las fuentes reportan el problema. El fondo a 0 conserva sus huecos, porque rodea
   a la maleza; partirlo llevaba el orto de prueba de 758 a 1311 polígonos. **Sin verificar** si ese
   fondo con huecos da problemas en el monitor. Los cortes quedan como líneas rectas entre polígonos de la misma dosis. Sin verificar en máquina si
   esas uniones afectan algo. La variante "límite" está implementada solo para Pro 700 / IntelliView, con la
   maleza como límite (no el complemento), copiando el archivo de Avansys.
8. **Salida empaquetada:** `<lote>_<monitor>/` con la estructura del pendrive y el mismo contenido en un
   .zip al lado.
9. Nombres de archivo cortos y ASCII (`safe_name`).

Opción agregada a pedido del usuario el 2026-11-06: **grilla cuadrada** de la zona de maleza (por defecto
apagada; 5 m como en los casos publicados de Avansys). Sin probar en máquina.

Criterios propios, no documentados por los fabricantes: Envizio Pro, Abelardo Cuffia y Microelec exportan
a la raíz porque su carpeta está **sin verificar**. AG Fusion Pro usa el máximo de AG Fusion (100), **sin
verificar**. TeeJet exporta TASKDATA para importar en Fieldware Link.

## Pendientes de validar en máquina o con el fabricante

- [ ] Formato y carpeta de Sensor, Caimán y Metalfor
- [ ] Si Pro 700 / IntelliView IV corta secciones con una dosis 0 dentro de la Rx
- [ ] Columnas exactas para la Rx on/off en Trimble Precision IQ
- [ ] Si Ag Leader InCommand con el firmware actual corta secciones en manchones chicos
- [ ] Qué marcas aceptan la "dosis fuera de la maleza" según Avansys
- [ ] Si el ISOXML de `export_isoxml` lo leen IntelliView IV / Pro 700 (según VRAFY, aceptan ISO/XML
      exportado desde SMS)

Antes de usar en campo, probar siempre con agua y un archivo chico.

## Fuentes

- FieldView Argentina – Prescripciones por monitor (PDF): https://climate.com/content/dam/climate/es-ar/website/images/prescripciones/Prescripciones%20por%20monitor.pdf
- VRAFY – Common Display Prescription Formatting Guide: https://vrafy.helpjuice.com/prescription-formatting/common-display-prescription-formatting-guide
- VRAFY – Case Pro700 / NH Intelliview 4: https://vrafy.helpjuice.com/prescription-formatting/cnh-v30-newer-prescriptions
- Case IH – Shapefile Import for AFS Pro 700 (PDF): https://processes.premiercrop.com/hubfs/Case%20IH%20AFS%20Pro%20700%20Shapefile%20Import.pdf?hsLang=en
- Scribe – Format Prescriptions for Pro 700 Shapefile Import: https://scribehow.com/shared/Format_Prescriptions_for_Pro_700_Shapefile_Import__N3a9yD3hQZa-bvSCckQIWQ
- Advantage Acre – Case IH Pro 700 v30: https://advantageacresupport.com/case-ih/
- Scribe – Raven Viper 4/4+: https://scribehow.com/shared/Format_Prescriptions_for_Raven_Viper_44__3lSgPD8zTRWEUwcV4L4vOw
- John Deere 4640 – Prescription Maps: https://displaysimulator.deere.com/onscreen_help/4640/current/en/mapping/map_rx.htm
- John Deere – Import Data (carpeta Rx): https://displaysimulator.deere.com/onscreen_help/S7combine/current/en/file_manager/file_manager_import_data.htm
- John Deere – Pla 120J (Gen 4 4640 de serie): https://www.deere.com/latin-america/es/pla/sprayer-pla120j/
- Hardi / Trimble (AgGPS\Prescriptions): https://hardi.com/download_file/eaa8c7b0-ba9c-4fd3-917d-eb313dbe7001/5878
- Stara – mapas por telemetría: https://www.stara.com.br/en/news/departamento-de-rede/how-to-send-variable-rate-maps-without-the-need-for-a-flash-drive
- TeeJet – VRA con Fieldware Link (PDF): https://www.teejet.com/es-es/-/media/dam/agricultural/usa/sales-material/technical-manual/98-05339-en-a4-r1-vra-applications.pdf
- Grainews – variable-rate spraying (JD / Raven y dosis 0): https://www.grainews.ca/features/am-i-set-up-for-variablerate-spraying
- Pix4D – AFS Pro 700 en pulverizadora Case: https://community.pix4d.com/t/importing-a-prescription-map-to-an-afs-pro-700-monitor-case-sprayer/34183
- Pix4D – Trimble GFX-750 weed map: https://community.pix4d.com/t/trimble-piq-gfx750-weed-prescription-map/33227 (y ?page=2)
- Pix4D – Ag Leader y manchones chicos: https://community.pix4d.com/t/weed-identification-in-season-spray-prescription-file-compatibility/25988
- AvanSys: https://avansys.com.ar/ · https://avansys.com.ar/guia-avansys-desktop · https://avansys.com.ar/guias
- AvanSys Desktop v1.0.34: https://github.com/AVANSYS-AGTECH-SOLUTIONS/avansys-desktop-update/releases/tag/v1.0.34
- Punto Biz – nota sobre Avansys: https://puntobiz.com.ar/negocios/crearon-una-app-que-detecta-problemas-en-estadios-tempranos-y-ahorra-costos-20251210600
- Documento completo (Claude Docs): https://claude.ai/artifact/Hct4KgmAUhDoVMDcsWq4LJ

## Detalle on/off: Case IH Pro 700 / NH IntelliView IV y Trimble GFX-750 (relevamiento 2026-11-06)

En ninguno de los dos monitores la dosis 0 dentro de la Rx garantiza por sí sola el corte de secciones.
Hay que usar el mecanismo de corte que documenta cada fabricante.

### Trimble GFX-750 / GFX-350/1060/1260 (Precision-IQ)

- Según Trimble, desde PiQ 14.60 hay un ajuste que crea **zonas de exclusión a partir de las zonas
  de la Rx con dosis 0**. Las zonas de exclusión cortan por sección, igual que los límites. Nombre exacto
  del menú: **sin verificar**. En 14.6 estaba en Test Pilot y hay que activar Test Pilot para cambiar las
  preferencias de Application Control.
- Alternativa documentada en la guía de implementos de PiQ: *Settings > Application Control > Section
  Control > "Turn Off All Sections on Zero Target Rate"*. Cierra **todas** las secciones cuando el
  objetivo es 0, así que el corte es de botalón completo, no por sección.
- *Settings > Application Control > Prescriptions > "Enable Prescription Usage Automatically"*. Si está
  apagado, hay que tocar el ícono RX rate a mano.
- Requiere la licencia de prescripciones (Field-IQ: 96553-07). Carpeta: `AgData/Prescriptions/`.
- Archivo: cubrir todo el lote, con la maleza a dosis X y el resto **como polígonos explícitos a 0**
  (el ajuste trabaja sobre zonas con dosis 0). Usuarios reportan que las zonas de aplicación con huecos
  (anillos) o muy fragmentadas no encienden las secciones. Con 1 L/ha en lugar de 0, las secciones
  quedan encendidas. Un usuario reportó lo mismo con PiQ 15.2.
- Workarounds no confirmados: cargar el área sin maleza como límite (`AgGPS/Data/<productor>/<lote>`,
  shp + .pos desde SMS); cargarla como cobertura previa con huecos en la maleza.

### Case IH Pro 700 / NH IntelliView IV

- El corte automático de secciones (AccuBoom en Patriot) es por **solape** (overlap) y por **límite**
  (boundary). Según un concesionario, también corta al entrar en un área interior marcada, como un
  canal. Se configura en *Toolbox > Overlap*: Overlap Control, Boundary Control, % fuera de límite,
  % de solape, start early/stop late. Requiere modo (A).
- La Rx entra por *Gestión de datos > Import2* (Source Shapefile, Data Type Prescription) y se asigna
  en *Performance > Rx Setup*. Fijar "Out of Zone Rate".
- *Toolbox > Product*: la dosis mínima y máxima limitan la dosis: "Rate will not be able to be adjusted
  outside of this range". Con mínimo > 0, una Rx en 0 se aplica a la dosis mínima.
- No se encontró documentación de CNH que diga que una dosis 0 en la Rx corta secciones. Un usuario logró
  que el monitor leyera 0 y 35–40 L/ha, pero las transiciones no funcionaron (zonas de 10×10 a 40×40 m).
  Para Trimble, Pix4D comenta que existe un formato "para Case On/Off", sin documentación pública.
- Pulverizadoras de arrastre Case: las ECU de fábrica pueden no admitir corte de secciones ni dosis
  variable (limitación de la máquina, no del monitor).

### Reglas de armado que se desprenden

1. Cubrir todo el lote: maleza a dosis X y resto a 0 explícito (`background=True`).
2. Exportar la zona de maleza sin anillos: unir los manchones con buffer o partir los huecos.
3. Agrandar y unir los manchones según la distancia recorrida durante el retardo de producto. El Pro 700
   usa 2,0 s por defecto, y la Patriot recomienda 0,7–2,0 s. Por ejemplo, a 20 km/h, 2 s son unos 11 m.
4. Variante "límite": en IntelliView IV funciona cargar la **zona de maleza** como límite (archivo de Avansys,
   ver arriba). Para otros monitores que cortan por límite, **pendiente de probar**.
5. Probar siempre con agua antes de cargar producto.

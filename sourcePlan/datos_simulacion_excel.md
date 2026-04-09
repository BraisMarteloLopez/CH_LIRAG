# Datos completos de simulacion -- Excel Lanchester_CIO_v1_4_.xlsm

Extraccion exhaustiva de todos los parametros, tablas, formulas y logica de calculo
usados en la simulacion. Fuente unica de verdad.

---

## 1. ESTRUCTURA GENERAL DEL LIBRO

Hojas relevantes:
- `Usuario`: Interfaz principal. Configuracion de combates (hasta 3), series temporales de Euler, resultados.
- `Datos`: Tablas de referencia (Estado tactico, Movilidad tactica por terreno).
- `Modificacion_datos`: Parametros por defecto de vehiculos (PROPIAS y ENEMIGAS) con sobreescritura de usuario. Logica IF(USUARIO="",DEFECTO,USUARIO).
- `BBDD_U_PROPIAS`: Base de datos de unidades propias. Calculo de tasas de destruccion agregadas.
- `BBDD_U_EN`: Base de datos de unidades enemigas. Misma estructura que PROPIAS.
- `BBDD_U_PROPIAS_comb_2`, `BBDD_U_PROPIAS_comb_3`: Replicas para 2do y 3er combate.
- `BBDD_U_EN_comb_2`, `BBDD_U_EN_comb_3`: Replicas para 2do y 3er combate.
- `CARGA DATOS`: Vacia (posiblemente gestionada por VBA).
- `Hoja1`: Vacia.

Rangos nombrados:
- `Estado`: Datos!$B$3:$E$10
- `Movilidad_tactica`: Datos!$B$12:$F$15
- `Variables_propias`: BBDD_U_PROPIAS!$B$5:$P$19
- `Variables_enemigo`: BBDD_U_EN!$B$6:$P$26
- `Combate_euler`: Usuario!$S$12:$FEN$18 (serie temporal 1er combate)
- `Combate_euler_2`: Usuario!$S$65:$FEN$71 (serie temporal 2do combate)
- `Combate_euler_3`: Usuario!$S$118:$FEN$124 (serie temporal 3er combate)

---

## 2. TABLA DE ESTADO TACTICO (Datos!B3:E10)

Rango nombrado: `Estado`
Usado por VLOOKUP en Usuario para obtener multiplicadores segun la situacion tactica.

| Estado | Mult. Fuerzas Propias (col C) | Mult. Fuerzas Opuestas (col D) | Movilidad (col E) |
|--------|-------------------------------|--------------------------------|--------------------|
| Retardo | 1 | 1/36 = 0.02778 | 1 |
| Ataque a una posicion defensiva | 1 | 1 | 1 |
| Busqueda del contacto | 0.9 | 1 | 1 |
| En posicion de tiro | 1 | 0.9 | 0 |
| Defensiva condiciones minimas de defensa | 1 | 1/2.25^2 = 0.19753 | 0 |
| Defensiva organizacion ligera | 1 | 1/2.75^2 = 0.13223 | 0 |
| Defensiva organizacion media | 1 | 1/4.25^2 = 0.05536 | 0 |
| Retrocede | 0.9 | 1 | 1 |

Formulas originales de columna D:
- Retardo: =1/6^2
- Def. condiciones minimas: =1/2.25^2
- Def. organizacion ligera: =1/2.75^2
- Def. organizacion media: =1/4.25^2

---

## 3. TABLA DE MOVILIDAD TACTICA (Datos!B12:F15)

Rango nombrado: `Movilidad_tactica`
Unidades no especificadas explicitamente (posiblemente km/h o factor adimensional).

| Terreno \ Movilidad | MUY_ALTA | ALTA | MEDIA | BAJA (PIE) |
|----------------------|----------|------|-------|------------|
| FACIL | Sin datos | 5 | 4 | 2.5 |
| MEDIO | Sin datos | 2.5 | 2.3 | 2 |
| DIFICIL | Sin datos | 1.5 | 1.3 | 1 |

Nota: La columna MUY_ALTA (C13:C15) contiene "Sin datos" en las tres filas.
Esta tabla se replica identica en filas 50-53 y 104-107 de la hoja Datos (para combates 2 y 3).
La etiqueta de la ultima columna varia: "BAJA" en la primera tabla, "PIE" en las replicas.

Rangos de validacion (desde Modificacion_datos):
- MUY_ALTA: [0-25]
- ALTA: [0-15]
- MEDIA: [0-15]
- BAJA: [0-15]

---

## 4. BASE DE DATOS DE VEHICULOS -- FUERZAS PROPIAS (Modificacion_datos + BBDD_U_PROPIAS)

### 4.1 Parametros convencionales

| Vehiculo | DUREZA | POTENCIA | PUNTERIA | CADENCIA | ALCANCE_MAX | FACTOR_DIST | CAP_C/C | POT_C/C | DUREZA_C/C |
|----------|--------|----------|----------|----------|-------------|-------------|---------|---------|------------|
| INFANTERIA_LIGERA | 20 | 10 | 0.2 | 5 | 600 | 0.6 | 1 | 300 | 10000 |
| LEOPARDO_2E | 700 | 690 | 0.75 | 2 | 4000 | 0.1 | 0 | - | 700 |
| LEOPARD_2A4 | 600 | 600 | 0.75 | 2 | 3500 | 0.1 | 0 | - | 600 |
| CENTAURO | 250 | 400 | 0.75 | 2 | 4000 | 0.15 | 0 | - | 250 |
| VEC | 75 | 60 | 0.6 | 3 | 1800 | 0.25 | 0 | - | 250 |
| INFANTERIA_BMR | 75 | 20 | 0.5 | 3.5 | 1500 | 0.4 | 0 | - | 250 |
| INFANTERIA_TOA | 75 | 20 | 0.5 | 3.5 | 1500 | 0.4 | 0 | - | 250 |
| TOA_SPIKE_I | 75 | 8 | 0.4 | 3.5 | 600 | 0.6 | 1 | 1000 | 75 |
| VAMTAC | 50 | 20 | 0.5 | 3.5 | 1500 | 0.4 | 0 | - | - |
| VAMTAC_SPIKE_I | 50 | 8 | 0.4 | 3.5 | 600 | 0.6 | 1 | 1000 | 50 |
| DRAGON_AT | 275 | 110 | 0.75 | 3 | 3000 | 0.4 | 1 | 1200 | 200 |
| VCI_PIZARRO | 275 | 110 | 0.7 | 3 | 3000 | 0.4 | 0 | - | 250 |
| DRAGON | 275 | 110 | 0.75 | 3 | 3000 | 0.4 | 0 | - | 200 |

### 4.2 Parametros C/C adicionales (solo vehiculos con CAPACIDAD_C/C = 1)

| Vehiculo | CADENCIA_C/C | ALCANCE_C/C | MUNICION | FACTOR_DIST_C/C |
|----------|-------------|-------------|----------|-----------------|
| INFANTERIA_LIGERA | 1 | 600 | 2 | 1 |
| TOA_SPIKE_I | 0.5 | 4000 | 6 | 0 (nota: ver ALC_ARM_PPAL=4000) |
| VAMTAC_SPIKE_I | 0.5 | 4000 | 6 | 0 |
| DRAGON_AT | 0.5 | 5500 | 4 | 0 |

Rangos de validacion por parametro:
- DUREZA: [1-2000]
- POTENCIA: [1-2000]
- PUNTERIA: [0.2-1]
- CADENCIA: [0.1-10]
- ALCANCE_MAX: [200-8000]
- FACTOR_DISTANCIA: [0-2]
- CAPACIDAD_C/C: {0, 1}
- POTENCIA_C/C: [1-2000]
- DUREZA_C/C: [1-15000]
- CADENCIA_C/C: [0.1-10]
- ALCANCE_C/C: [200-8000]
- MUNICION: [1-20]
- FACTOR_DISTANCIA_C/C: [0-2]
- ALCANCE_ARM_PPAL: [200-8000]

---

## 5. BASE DE DATOS DE VEHICULOS -- FUERZAS ENEMIGAS (Modificacion_datos + BBDD_U_EN)

### 5.1 Con capacidad C/C

| Vehiculo | DUREZA | POTENCIA | PUNTERIA | CADENCIA | ALC_MAX | FACT_DIST | CAP_C/C | POT_C/C | DUREZA_C/C | CAD_C/C | ALC_C/C | MUNIC |
|----------|--------|----------|----------|----------|---------|-----------|---------|---------|------------|---------|---------|-------|
| INFANTERIA_LIGERA | 10 | 20 | 0.2 | 5 | 600 | 0.6 | 1 | 300 | 10000 | 1 | 600 | 2 |
| T-90A_AT | 570 | 630 | 0.75 | 1.7 | 3500 | 0.1 | 1 | 700 | 650 | 0.3 | 5000 | 6 |
| T-80U_AT | 500 | 600 | 0.75 | 1.7 | 3500 | 0.15 | 1 | 700 | 550 | 0.3 | 5000 | 6 |
| T-72BM_AT | 490 | 600 | 0.75 | 1.7 | 3000 | 0.2 | 1 | 700 | 540 | 0.3 | 4000 | 4 |
| BMP-3_AT | 175 | 50 | 0.6 | 2.6 | 2500 | 0.5 | 1 | 750 | 175 | 0.4 | 5000 | 4 |
| BMP-2_AT | 130 | 50 | 0.5 | 2.6 | 2000 | 0.5 | 1 | 925 | 250 | 0.4 | 4000 | 4 |
| BMP-2M_AT | 130 | 50 | 0.5 | 2.6 | 2000 | 0.5 | 1 | 1100 | 250 | 0.4 | 5500 | 6 |
| BMP-1P_AT | 130 | 550 | 0.5 | 1.6 | 1300 | 0.5 | 1 | 480 | 130 | 0.4 | 2000 | 4 |
| BMD-3_AT | 65 | 50 | 0.9 | 2.6 | 1200 | 0.45 | 1 | 925 | 65 | 0.4 | 4000 | 4 |
| BDRM-2(AT-5) | 60 | 8 | 0.4 | 3.5 | 600 | 0.6 | 1 | 925 | 250 | 0.6 | 4000 | 10 |

### 5.2 Sin capacidad C/C

| Vehiculo | DUREZA | POTENCIA | PUNTERIA | CADENCIA | ALC_MAX | FACT_DIST | DUREZA_C/C |
|----------|--------|----------|----------|----------|---------|-----------|------------|
| BTR-80A | 130 | 50 | 0.5 | 3 | 2000 | 0.5 | 200 |
| BDRM_reco | 60 | 20 | 0.5 | 3.5 | 1500 | 0.4 | 250 |
| T-90 A | 570 | 630 | 0.75 | 2 | 3500 | 0.1 | 650 |
| T-80U | 500 | 600 | 0.75 | 2 | 3500 | 0.15 | 550 |
| T-72 BM | 490 | 600 | 0.75 | 2 | 3000 | 0.2 | 540 |
| BMP-3 | 175 | 50 | 0.6 | 3 | 2500 | 0.5 | 175 |
| BMP-2 | 130 | 50 | 0.5 | 3 | 2000 | 0.5 | 250 |
| BMP-2M | 130 | 50 | 0.5 | 3 | 2000 | 0.5 | 250 |
| BMP-1P | 130 | 550 | 0.5 | 2 | 1300 | 0.5 | 130 |
| BMD-3 | 65 | 50 | 0.9 | 3 | 1200 | 0.45 | 65 |

Nota: Varios vehiculos enemigos existen en version _AT (con C/C) y sin sufijo (sin C/C), con cadencias distintas.

---

## 6. LOGICA DE SOBREESCRITURA DE PARAMETROS

Hoja `Modificacion_datos` columnas AL-AY: formula tipo `=IF(USUARIO="",DEFECTO,USUARIO)`

Para cada parametro de cada vehiculo, si el usuario ha introducido un valor en la columna USUARIO, se usa ese valor. Si la celda USUARIO esta vacia, se usa el valor por DEFECTO. Esto permite al usuario personalizar parametros sin alterar los valores base.

---

## 7. CALCULO DE TASA DE DESTRUCCION CONVENCIONAL (S_conv)

Hoja: BBDD_U_PROPIAS (y BBDD_U_EN, misma logica)

### 7.1 Probabilidad de inoperativo al impactar (T)
```
T = 1 / (1 + EXP((D_enemigo - P_propio) / 175))
```
Donde:
- P_propio = Potencia de fuego agregada ponderada (SUMPRODUCT vehiculos * potencia / total vehiculos)
- D_enemigo = Dureza agregada ponderada del enemigo

### 7.2 Degradacion del disparo por distancia (g)
```
g = -0.188*(d/1000) - 0.865*f + 0.018*(d/1000)^2 - 0.162*(d/1000)*f + 0.755*f^2 + 1.295
```
Donde:
- d = Distancia de enfrentamiento (m)
- f = Factor de distancia agregado ponderado
- Si d > Alcance_max: g = 0

Degradacion acotada: G = max(0, min(1, g))

### 7.3 Tasa de destruccion convencional
```
S_conv = T * G * U * c
```
Donde:
- T = Probabilidad de inoperativo
- G = Degradacion acotada por distancia
- U = Punteria agregada ponderada
- c = Cadencia agregada ponderada

---

## 8. CALCULO DE TASA DE DESTRUCCION C/C ESTATICA (S_cc)

Misma hoja BBDD_U_PROPIAS / BBDD_U_EN.

### 8.1 Probabilidad inoperativo C/C (T_cc)
```
T_cc = 1 / (1 + EXP((D_cc_enemigo - P_cc_propio) / 175))
```

### 8.2 Degradacion disparo C/C (g_cc)
Misma formula polinomica que la convencional pero con parametros C/C:
```
g_cc = -0.188*(d/1000) - 0.865*f_cc + 0.018*(d/1000)^2 - 0.162*(d/1000)*f_cc + 0.755*f_cc^2 + 1.295
```
G_cc = max(0, min(1, g_cc))

### 8.3 Tasa C/C estatica
```
S_cc = c_cc * T_cc * G_cc
```
Nota: No incluye punteria (U) en la formula C/C, solo cadencia C/C.

---

## 9. AGREGACION DE PARAMETROS (media ponderada por n_vehiculos)

Para cada parametro X, el valor agregado es:
```
X_agregado = SUMPRODUCT(n_vehiculos_i, X_i) / SUM(n_vehiculos_i)
```
Implementado como: `=IFERROR(SUMPRODUCT($S$4:$S$13, X4:X13) / $S$14, 0)`

Excepcion: ALCANCE_C/C y FACTOR_DISTANCIA_C/C usan ponderacion adicional por el producto (n_vehiculos * MUNICION_C/C).

---

## 10. CONFIGURACION DE COMBATE (Hoja Usuario)

### 10.1 Parametros de entrada por combate

Para cada combate (1, 2, 3) se configuran:

- **TERRENO**: FACIL / MEDIO / DIFICIL (celda C3)
- **MOVILIDAD TACTICA**: MUY_ALTA / ALTA / MEDIA / BAJA (celda F5)
- **ESTADO tactico AZUL**: Seleccion de la tabla Estado (celda F6)
- **ESTADO tactico ROJO**: Mismo (celda F25)
- **Tipo y cantidad de vehiculos AZUL**: hasta 10 tipos (B7:C16)
- **Tipo y cantidad de vehiculos ROJO**: hasta 10 tipos (B26:C35)
- **Distancia de enfrentamiento**: por defecto y usuario (D21/F21), en metros
- **Proporcion de fuerzas empenadas**: por defecto 2/3 para 1er combate, 1 para 2do y 3ro (G15, G34)
- **Factor arbitrario tasa de destruccion**: [0, 2], por defecto 1 (H8, H27)
- **Factor arbitrario n_vehiculos**: [0, 2], por defecto 1 (H9, H28)
- **AFT recibidas**: numero de AFT (F12, F31)
- **Efecto AFT en % de bajas**: fraccion (F13, F32)
- **Refuerzos** (solo combates 2 y 3): tipos y cantidades adicionales

### 10.2 Multiplicadores de estado

Obtenidos via VLOOKUP sobre la tabla Estado:
```
Multiplicador_propio = VLOOKUP(Estado_seleccionado, Estado, 2, FALSE)
Multiplicador_oponente = VLOOKUP(Estado_seleccionado, Estado, 3, FALSE)
```

### 10.3 Tasas de destruccion finales usadas en simulacion

```
Tasa_final_azul = S_conv_azul * Factor_arbitrario * Mult_propio_azul * Mult_oponente_rojo
Tasa_final_roja = S_conv_rojo * Factor_arbitrario * Mult_propio_rojo * Mult_oponente_azul
```

Formulas reales (ejemplo 1er combate):
- G18 (tasa conv. final azul): `=D18 * H8 * G6 * H25`
  donde D18=S_conv_azul, H8=Factor_arbitrario, G6=Mult_propio_azul, H25=Mult_oponente_rojo
- H18 (tasa cc final azul): `=E18 * G6 * H8 * H25`
  donde E18=S_cc_azul

---

## 11. MOTOR DE SIMULACION -- METODO DE EULER

### 11.1 Parametros del integrador

- Salto temporal h: `=1/600` horas (=0.1 minutos = 6 segundos) (celda S11)
- Rango de iteraciones: columnas S a FEN (aprox. 4200 columnas = ~420 minutos de combate simulado)

### 11.2 Ecuaciones diferenciales (Lanchester)

Filas 14-15 de la serie temporal:

**AZUL (fila 14):**
```
AZUL(t+h) = max(0, AZUL(t) - ROJO(t) * (S_cc_rojo + S_conv_final_rojo) * h)
```
Formula: `=IF(T14 - T15*(T17+$G$37)*$S$11 < 0, 0, T14 - T15*(T17+$G$37)*$S$11)`

**ROJO (fila 15):**
```
ROJO(t+h) = max(0, ROJO(t) - AZUL(t) * (S_cc_azul + S_conv_final_azul) * h)
```
Formula: `=IF(T15 - T14*(T16+$G$18)*$S$11 < 0, 0, T15 - T14*(T16+$G$18)*$S$11)`

### 11.3 Tasa C/C variable con municion (fila 16 y 17)

La tasa C/C no es constante: disminuye a medida que se consume la municion.

```
S_cc_azul(t) = IF(Municion_total=0, 0,
               IF(S_cc_base * (Municion_total - Cadencia_cc * t) / Municion_total < 0, 0,
                  S_cc_base * (Municion_total - Cadencia_cc * t) / Municion_total))
```
Formula real: `=IF($R16=0, 0, IF($H18*(Usuario!$R16 - BBDD_U_PROPIAS!$AC$14*T13)/$R16 < 0, 0, $H18*($R16 - BBDD_U_PROPIAS!$AC$14*T13)/$R16))`

Donde:
- $R16 = Municion total azul (BBDD_U_PROPIAS!AE14)
- BBDD_U_PROPIAS!$AC$14 = Cadencia C/C agregada
- T13 = tiempo transcurrido
- $H18 = Tasa S_cc base

### 11.4 Condiciones iniciales

- AZUL(0) = Vehiculos empenados = n_vehiculos_total * Proporcion_empenada
- ROJO(0) = Vehiculos empenados enemigos = n_vehiculos_total_rojo * Proporcion_empenada

### 11.5 Deteccion de fin de combate (fila 18)

```
=IF(AND(AZUL=0, ROJO=0), 0.5, IF(AZUL=0, -1, IF(ROJO=0, 1, 0)))
```
- 1 = Victoria azul
- -1 = Victoria roja
- 0.5 = Empate (ambos destruidos)
- 0 = Combate continua

---

## 12. CALCULO DE VENTAJA ESTATICA

```
Proporcion_estatica = (S_conv_azul + S_cc_azul) * N_azul^2 / ((S_conv_rojo + S_cc_rojo) * N_rojo^2)
```
Formula real (S28): `=IF(G37+H37=0, 10, IF(G18+H18=0, -10, (G18+H18)*F18^2 / ((G37+H37)*F37^2)))`

Ventaja estatica (L20): Calculo numerico del ratio.
Vencedor (K21): Determinado por el signo/valor.

---

## 13. RESULTADOS POR COMBATE

Cada combate produce:
- Vehiculos operativos tras el combate (AZUL y ROJO): vehiculos restantes
- Duracion del combate (minutos)
- % de bajas sufrido por cada bando
- Vencedor
- Vehiculos empenados vs totales

Los vehiculos supervivientes del combate N pasan como entrada al combate N+1, sumandose a los refuerzos.

---

## 14. EFECTO DEL TERRENO EN LA SIMULACION

Segun la evidencia del Excel, el terreno SOLO afecta a la MOVILIDAD TACTICA (tabla Datos!B12:F15).
El terreno NO tiene efecto directo sobre:
- La tasa de destruccion convencional (S_conv)
- La tasa de destruccion C/C (S_cc)
- Ningun multiplicador de efectividad de fuego

La movilidad tactica se selecciona en Usuario (celda F5) pero su uso concreto en las ecuaciones de Lanchester no es visible en las formulas extraidas. La columna E de la tabla Estado indica si hay movilidad (1) o no (0) segun el estado tactico, pero no se ha identificado una formula que use directamente el valor numerico de velocidad de la tabla Movilidad_tactica en el calculo de bajas.

---

## 15. EFECTO DE LAS AFT (Armas de Fuego de Tiro indirecto / Apoyo de Fuego)

Las AFT se aplican como bajas proporcionales pre-combate o durante el combate:
- AFT recibidas: numero de AFT (configurado por usuario)
- Efecto de las AFT: fraccion de bajas (ej. 0.025 = 2.5%)
- Formula (ejemplo): `=$V$23 * C7 / $C$18` para vehiculos perdidos por AFT
- Bajas AFT = n_vehiculos * Efecto_AFT

---

## 16. CONTRASENA

Celda B1 de hoja Datos: CONTRASENA_DESBLOQUEAR = "CIO"

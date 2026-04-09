**LanchesterAPP**

Documentación sobre el modelo Lancaster propuesto por el cliente para
mejorar la métrica actual de comprobación de potencias relativas del
documento de referencia interno al teorizar la asignación de fuerzas en
un combate.

## La base de datos

Para cada _vehículo_ se incluyen una serie de propiedades en
general y para C/C (combate contracarro)

> **Dureza (D), Potencia (P), Puntería (U), Cadencia (c), Alcance Máximo
> ($A^{MAX}$)** y **Factor distancia (f)**

El factor distancia es la pérdida de precisión de tiro en función de la
distancia.

Además, hay dos propiedades únicas para contracarro: **munición (M)** y
**capacidad**. Creo que capacidad no se usa.

Además, _cada equipo_ tiene tres _variables adiciones_. Cada una equivale a un número.

-   **Movilidad Táctica** (4 opciones): MUY ALTA, ALTA, MEDIA, BAJA

-   **Terreno** (3 opciones): FÁCIL, MEDIO, DIFICIL

-   **Situación** (8 opciones): mirar *Modificación_datos en el Excel*

## Introducción de variables

Se introducen los **carros a combatir** en cada unidad y el **número
(V)** de carros.

También la **distancia inicial (d)** del combate.

Variables adicionales 1

> Las variables adicionales indicadas antes (Movilidad táctica, Terreno
> y Situación) quizás deberían poder seleccionarse en configuración
> avanzada.
>
> Variables adicionales 2
>
> Estás variables sí que deberían seleccionarse en configuración
> avanzada:

-   **Proporción**: por defecto 2/3, el número de fuerzas efectivas que
    participaran en este combate. (1)

-   **Acciones de Fuego Tipo (AFT)**: es un porcentaje, que elimina una
    parte de los vehículos iniciales asignados a la misión (se aplica
    después de proporción).

Con esto el número de vehículos final queda
$V : = proporción \cdot (100\% - AFT) \cdot V$.

-   **Factor arbitrario de tasa de destrucción (λ)**: multiplica por un
    número la tasa de destrucción. Se supone que permite ponderar
    variables no cuantificables como la *moral*.

> Algunas variables incluidas en el Excel las vamos a descartar por el
> momento (2).

Todas estas variables, excepto **distancia** y **Terreno**, se
establecen para el equipo propio, llamado Azul (A), y para el equipo
enemigo, llamado Rojo (R). Del mismo modo, las variables derivadas se
calculan para cada bando.

## Variables derivadas

Todas las variables de cada vehículo se agrupan mediante *media
ponderada*, dejando una variable de cada por equipo (3).

Definimos las siguientes variables para el equipo Azul, pero se definen
igual para el Rojo.

-   Probabilidad de inoperar del equipo Azul:

$$T_{A} = \ \frac{1}{1 + e^{(D_{R} - P_{A})/175}}$$

-   Degradación acotada de Azul:

$$G_{A} = - 0,188 \cdot d - 0,865 \cdot f_{A} + 0,018 \cdot d^{2} - 0,162 \cdot d \cdot f_{A} + 0,755 \cdot f_{A}^{2} + 1,295$$

Acotada en \[0,1\] si $d \leq A_{AZUL}^{MAX}$. En caso contrario es 0.

-   **Tasa de destrucción** Azul ($S_{A}$): (4)

$$S_{A} = \lambda_{A} \cdot (T_{A} \cdot U_{A} \cdot G_{A} \cdot c_{A})$$

-   Variables derivadas contracarro

    -   **Tasa de destrucción contracarro estática Azul (**$S_{A}^{cc})$

> $$S_{A}^{cc} = T_{A}^{cc} \cdot G_{A}^{cc} \cdot c_{A}^{cc}$$
>
> Con las variables calculadas como en la versión no contracarro.
>
> NUEVO: Acotada entre \[-10 y 10\].

-   **Proporción estática** (φ): es una medida derivada que permite
    evaluar el éxito de una misión a priori. Se define como

$$\varphi = \ \frac{\left( T_{A} + S_{A}^{cc} \right) \cdot V_{A}}{\left( T_{R} + S_{R}^{cc} \right) \cdot V_{R}}$$

acotada entre \[-10 y 10\].

¿La proporción estática puede dar valores negativos? Yo creo que hemos
mezclado proporción estática con Tasa de destrucción contracarro
$S^{cc}$, y seguramente no haga falta normalizar para dar la
probabilidad estática.

> La **probabilidad estática** ($\widetilde{\varphi}$) la definimos (5)
> normalizando $\varphi$

$$\widetilde{\varphi} = \ \frac{\varphi - 10}{20}$$

-   las **funciones** de tasa de **contracarro** $S_{R}^{T_{cc}}(t)$ y
    $S_{A}^{T_{cc}}(t)$ se definen

> $$S_{R}^{T_{cc}}(t) = r^{cc}(t) = \left\{ \begin{array}{r}
> \frac{S_{R}^{ccf} \cdot (M_{R} - c_{R} \cdot t)}{M_{R}}\ si\ S_{R}^{ccf} > 0\ y\ r^{cc}(t) > 0 \\
> 0\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ en\ otro\ caso
> \end{array} \right.$$
>
> $$S_{A}^{T_{cc}}(t) = a^{cc}(t) = \left\{ \begin{array}{r}
> \frac{S_{A}^{ccf} \cdot (M_{A} - c_{A} \cdot t)}{M_{A}}\ si\ S_{A}^{ccf} > 0\ y\ a^{cc}(t) > 0 \\
> 0\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ en\ otro\ caso
> \end{array} \right.$$
>
> ¿FALTA ALGO?

-   Movimiento (6)

> Esta sección no se incluye en el Word pero aparece adicional en el
> Excel.

-   **Velocidad** ($v_{A}$)

> Obtenemos una variable llamada velocidad, medida en km/h, para ambos
> equipos, tomada de una tabla Datos.Movilidad_tactica del Excel, y
> referenciada por las variables *Movilidad Táctica* y *Terreno*.
>
> Por ejemplo, si la Movilidad Táctica de Azul es ALTA y el Terreno se
> considera FÁCIL, la velocidad de Azul será
>
> $$v_{A} = TABLA[Movilidad\_táctica, Terreno] = 5$$

-   **Velocidad proporcional** ($v_{A}^{\varphi}$)

> La velocidad proporcional depende de la velocidad y de la proporción
> estática (7) y se define como
>
> $$v_{A}^{\varphi} = \left\{ \begin{matrix}
> v_{A} \cdot (0.1 \cdot (\varphi - 9) + 1) & si\ Situación\ permite\ movilidad \\
> 0 & en\ otro\ caso
> \end{matrix} \right.$$
>
> *Datos.Situación.movilidad* es una columna de la *Tabla Situación*
> (también llamada *Estado*) en la hoja *Datos*, que indica si una
> Situación de combate permite la movilidad o no. Es un booleano.
>
> $$v_{R}^{\varphi} = \left\{ \begin{matrix}
> v_{R} \cdot (0.1 \cdot \left( \frac{1}{\varphi} - 9 \right) + 1) & si\ Situación\ permite\ movilidad \\
> 0 & en\ otro\ caso
> \end{matrix} \right.$$
>
> En este caso se define $v_{R}^{\varphi}$ de manera diferente porque la
> proporción estática está definida como directa para Azul e inversa
> para Rojo.

-   Equipo más rápido

> Básicamente devuelve si el equipo Azul o Rojo es más rápido.
>
> $$\operatorname{máx}{\ (v_{A}^{\varphi},v_{R}^{\varphi})}$$

-   Tiempo de desplazamiento

> Devuelve el tiempo dedicado al desplazamiento de las unidades. (8).
>
> $$\frac{\frac{d}{1000} \cdot 60}{\operatorname{máx}{\ (v_{A}^{\varphi},v_{R}^{\varphi})}}$$

## La simulación (la aproximación de la solución Lancaster)

La variable $h\  \equiv \ \frac{1}{600}$ probablemente para indicar que
la simulación se calcula en incrementos de 0,1s.

Buscamos aproximar las ecuaciones de Lancaster (9) mediante el método de
Euler explicito (10).

$$a_{i + 1} = \left\{ \begin{matrix}
a_{i} - \ r_{i} \cdot \left( r_{i}^{cc} + T_{R} \right) \cdot h & si\ a_{i + 1} > 0\  \\
0 & si\ a_{i + 1} \leq 0
\end{matrix} \right.$$

$$r_{i + 1} = \left\{ \begin{matrix}
r_{i} - \ a_{i} \cdot \left( a_{i}^{cc} + T_{A} \right) \cdot h & si\ r_{i + 1} > 0\  \\
0 & si\ r_{i + 1} \leq 0
\end{matrix} \right.$$

Con $a_{0} = \ V_{A}$ y $r_{0} = \ V_{R}$, tras haberles aplicado las
variables de proporción y AFT como se describe en el apartado 2.

Además,
$r_{i}^{cc} = \ r^{cc}\left( t_{i} \right) = r^{cc}(i \cdot h)\ $y
$a_{i}^{cc} = \ a^{cc}\left( t_{i} \right) = a^{cc}(i \cdot h)$.

Y ejecutamos el sistema hasta que alguna de las dos variables llega a 0.

## ¿Qué devolvemos? 

Primero la **probabilidad estática** $\widetilde{\varphi}$, por lo que
no hace falta ni ejecutar la simulación.

Además, **mediante una** **petición del usuario** podremos mostrarle el
**bando ganador** tras la simulación, las **bajas** en cada bando y el
**tiempo del combate**. Además daremos una **gráfica** con la evolución
de las bajas en el combate.

También el **desplazamiento** de cada equipo y el **tiempo de
desplazamiento**.

*Figura 1. Ejemplo de gráfica esperada.*

Al devolver las bajas en cada bando, sumaremos lo que hemos restado
antes por proporción (11).

NO SE HA INCLUIDO INFORMACIÓN SOBRE ENFRENTAMIENTOS EN ESCALONES
(ASIGNAS LAS FUERZAS Y HACES VARIOS ENFRENTAMIENTOS) PERO IMAGINO QUE
SERÍA AÑADIR SIMPLEMENTE UN BUCLE A ESTA IMPLEMENTACIÓN).

## Notas 

(1) El mando no puede saber que tropas participarán en la batalla. Por
    tanto, en la simulación tomamos la proporción entre todas. Para más
    adelante se pueden hacer simulaciones seleccionando al azar que
    tropas de los 2/3 participan. (Método de MonteCarlo).

(2) El **factor arbitrario de vehículos**, que multiplica el número de
    vehículos, se descarta por el momento.

(3) Que Alcance Máximo sea la media ponderada de los Alcances me escama,
    porque hay carros que no podrán apuntar antes, y no me parece
    difícil añadir esta condición en la simulación. Aún así esto se deja
    para una mejora posterior

(4) En el Word aparece un parámetro adicional al definir la tasa de
    destrucción: $\alpha = \frac{1}{600}$. El analista postula que puede
    tratarse de una adecuación al tiempo, pues t se define en minutos y
    esto equivaldría a 0,1 segundos. En cualquier caso, no hemos que
    $\alpha$ se aplique como tal en el Excel. Puede tratarse del factor
    arbitrario de tasa de destrucción (λ) con un valor predefinido, o
    del h del método explícito de Euler, pues en el Excel se define con
    el mismo valor.

(5) El Excel original no ofrece ninguna probabilidad, solo una medida
    binaria que indica si gana el bando propio (Azul) o enemigo (Rojo).
    Así que esta parte la hemos inventado nosotros.

(6) La movilidad no se nombra específicamente en el documento, pero
    aparece también en el Excel, por eso la incluimos.

(7) Al no venir esta fórmula en el Excel no tengo claro de donde sale
    0.1, y sobre todo el 9 que se resta, el cual me preocupa porque
    parece un número arbitrario.

(8) Extrañamente, el tiempo de desplazamiento solo tiene en cuenta la
    velocidad del mejor equipo, cuando realmente hasta que los dos
    equipos no lleguen a línea de tiro creo van a seguir moviendose.

Además, las unidades del tiempo de desplazamiento son

> $$\frac{\frac{d\ m}{1000\ \frac{m}{km}} \cdot 60\frac{\min}{h}}{\operatorname{máx}{\ (v_{A}^{\varphi},v_{R}^{\varphi})}\frac{km}{h}} = \min$$

(9) Anexo 1

(1) Anexo 2

(2) Las bajas sufridas por AFT no se añaden porque ya se han restado
    COMPROBAR

## Anexos 

**Anexo 1: las ecuaciones de Lancaster modificadas**

Definimos en función del tiempo las unidades en combate no destruidas
del bando propio o azul (A(t)) y del bando enemigo ((R(t)).

Las ecuaciones diferenciales de Lancaster originales asumen que la
variación en las unidades de un bando se debe únicamente a las unidades
vivas del otro bando, multiplicadas por una constante conocida como tasa
de destrucción (S) (Anexo 3).

Las ecuaciones modificadas añaden una segunda dependencia, la tasa de
contracarro, que varía en función del tiempo y mide, entre otros, la
munición restante.

$$\left\{ \begin{array}{r}
\frac{\partial A(t)}{\partial t} = - \left[ S_{R} + S_{R}^{T_{cc}}(t) \right] \cdot R(t) \\
\frac{\partial R(t)}{\partial t} = - \left[ S_{A} + S_{A}^{T_{cc}}(t) \right] \cdot A(t)
\end{array} \right.$$

Como sabemos las unidades de cada bando al principio del combate, se
trata de un problema de valor inicial o problema de Cauchy, y el teorema
de Cauchy-Lipschitz garantiza que existe una única solución para estas
ecuaciones.

De las ecuaciones de Lancaster original (sin $S^{T_{cc}}(t)$) podemos
deducir que

$$S_{A} \cdot A^{2}(t) - S_{R} \cdot R^{2}(t) = n$$

Con n constante. Así que si $A(0) = n \cdot R(0)$, entonces
$S_{R} = n^{2} \cdot S_{A}$ para tener equilibrio.

Además, las ecuaciones de Lancaster original hay una solución directa y
sencilla ya que se trata de un problema tipo. Para las ecuaciones
modificadas se prefiere usar un método numérico para poder seguir
cambiándolas.

**Anexo 2: el método de Euler explicito**

Este apartado explica el método numérico empleado para aproximar el
valor de la ecuación diferencial Lancaster.

La derivada de una función en $\mathbb{R}$ puede escribirse como

$$x^{'}(t) = \ \lim_{h \rightarrow 0}\frac{x^{'}(t + h) - x'(t)}{h}$$

Y si definimos $f(x,t) \equiv x'(t)$, podemos ver la ecuación (con
límite en ambos lados) como

$$x(t + h) = h \cdot f(x,t) + x(t)$$

Así que definimos $x_{i + n} \equiv x(t + n \cdot h)$ para obtener la
sucesión

$$x_{i + 1} = h \cdot f\left( x_{i},t_{i} \right) + x_{i}$$

Que aproxima la derivada en todo el dominio de la función
$t \in [ a,b]$, definiendo $t_{i} \equiv \ i \cdot h$ y
$h = \left| [ a,b] \right| = \ \frac{b - a}{2}$, aunque en
la práctica podemos fijar h y elegir el dominio.

_Extra_

Si se toma $x_{i + 1} = h \cdot f\left( x_{i + 1},t_{i} \right) + x_{i}$
estamos en Euler implícito, donde debemos resolver una ecuación para
poder despejar $x_{i + 1}$.

h no tiene que estar idénticamente distribuido siempre que los h caigan
dentro de \[a, b\].

**Anexo 3: la base matemática de Lanchaster**

Este apartado desarrolla algunas nociones teóricas que no se incluyen en
la documentación oficial. No aplica al desarrollo, su único objetivo es
didáctico.

**A3.1. La definición del problema Lanchester original**

El problema original de Lancaster se define como

$$\left\{ \begin{array}{r}
\frac{\partial A(t)}{\partial t} = - S_{R} \cdot R(t) \\
\frac{\partial R(t)}{\partial t} = - S_{A} \cdot A(t)
\end{array} \right.$$

donde

$$\begin{matrix}
t:tiempo\ del\ transcurso\ del\ enfrentamiento \\
S_{A}:tasa\ de\ destrucción\ de\ Azul\ (unidades\ de\ Rojo\ que\ Azul\ destruye\ en\ una\ unidad\ de\ tiempo) \\
S_{R}:tasa\ de\ destrucción\ de\ Rojo\ (unidades\ de\ Azul\ que\ Rojo\ destruye\ en\ una\ unidad\ de\ tiempo)
\end{matrix}$$

$$\begin{matrix}
A(t):número\ de\ unidades\ de\ Azul\ en\ función\ del\ tiempo \\
R(t):número\ de\ unidades\ de\ Rojo\ en\ función\ del\ tiempo
\end{matrix}$$

**A3.2. Demostración de la ecuación cuadrática de Lanchester**

$S_{A} \cdot A^{2}(t) - S_{R} \cdot R^{2}(t) = n$, con $n$ constante.
La demostración es la siguiente

$$\frac{A^{'}}{R^{'}} = \frac{- S_{R} \cdot R}{- S_{A} \cdot A}\  \Longleftrightarrow S_{A} \cdot A \cdot A^{'} = S_{R} \cdot R \cdot R^{'} \Rightarrow \ \frac{1}{2} \cdot \frac{\partial}{\partial t}\left( S_{A} \cdot A^{2} \right) = \ \frac{1}{2} \cdot \frac{\partial}{\partial t}\left( S_{R} \cdot R^{2} \right) \Leftrightarrow$$

$$\Leftrightarrow \ \frac{\partial}{\partial t}\left( S_{A} \cdot A^{2} - S_{R} \cdot R^{2} \right)\  \Leftrightarrow S_{A} \cdot A^{2} - S_{R} \cdot R^{2} = cte$$

Por tanto, si $A(0) = n \cdot R(0)$, entonces
$S_{R} = n^{2} \cdot S_{A}$ para llegar a un equilibrio.

**A3.3. Demostración de la ecuación cuadrática para nuestro problema**

Es mas complicado porque $S_{R}^{T_{cc}}$ es dependiente de t. INTENTAR
HACER

**A3.4. Definiciones matemáticas para garantizar la solución**

-   **Problema de valor inicial o de Cauchy**

> Un problema se considera de valor inicial si está definido como
>
> $$\begin{matrix}
> x^{'}(t) = f\left( t,\ x(t) \right),\ t \in I = [a,b] \subset \mathbb{R} \\
> x\left( t_{0} \right) = \ \xi_{0}
> \end{matrix}$$

-   **Función autónoma**

> Si $f$ no depende de $t$ se define autónoma. El problema de Lanchester
> original es autónomo pero nuestra modificación no porque
> $S_{R}^{T_{cc}}$ depende directamente de $t$.

-   **Teorema de Cauchy-Lipschitz, también llamado de Picard-Lindelöf**

> Si f es una función continua sobre el rectangulo R y es lipschitziana
> en la segunda variable, entonces existe una única solución x(t) del
> problema de valor inicial definida en
> $[t_{0} - h, t_{0} + h]$, siendo h la menor longitud de
> R.

-   **Función lipschitziana en la segunda variable**

> $$\exists L > 0,,\ \left| \left| f(t,\xi) - f(t,\eta) \right| \right| \leq L \cdot ||\xi - \eta||$$
>
> Me parece una función doblemente derivable ($f \in \mathcal{C}^{2}$)
> es lipschitziana.

**A3.4. Solución directa de las ecuaciones diferenciales de Lanchester**

Si derivamos en una ecuación de Lanchaster y sustituimos con la otra,
obtenemos que

$$\frac{\partial^{2}A}{\partial t^{2}} = (S_{A} \cdot S_{R}) \cdot A$$

La cual es un tipo de función del tipo $A^{''} = k^{2}A$ llamadas
oscilador armónico. Estas ecuaciones tienen como solución

$$A(t) = c_{1}e^{\sqrt{(S_{A} S_{R}) \cdot t}} + c_{2}e^{-\sqrt{(S_{A} S_{R}) \cdot t}}$$

Sabiendo que
$A_{0} : = A(0)=c_{1} + c_{2}$
y
$-S_{R} \cdot R_{0}\ =\left. \ \frac{\partial R(t)}{\partial t} \right]_{t = 0}=k \cdot c_{1} + k \cdot c_{2}$

Podemos hacer un sistema de ecuaciones que nos devuelve la solución

$$A(t) = \frac{1}{2}\left[ \left( A_{0} - R_{0}\sqrt{\frac{S_{R}}{S_{A}}} \right)e^{t\sqrt{S_{R}S_{A}}} + \left( A_{0} - R_{0}\sqrt{\frac{S_{R}}{S_{A}}} \right)e^{- t\sqrt{S_{R}S_{A}}} \right]$$

$$R(t) = \frac{1}{2}\left[ \left( R_{0} - A_{0}\sqrt{\frac{S_{A}}{S_{R}}} \right)e^{t\sqrt{S_{A}S_{R}}} + \left( R_{0} - A_{0}\sqrt{\frac{S_{A}}{S_{R}}} \right)e^{- t\sqrt{S_{A}S_{R}}} \right]$$

**A3.5. El tiempo en función de los vehículos**

El documento original del cliente nos ofrece la función inversa de
calcula el tiempo en que estamos en función del número de vehículos que
nos quedan ($t(A$)). Dejamos este apartado para señalar que es posible
calcular esa función, aunque en este punto del desarrollo no le vemos
utilidad. En el Excel aparece esta función, pero su implementación da
error y el resto del código no se resiente.

**A3.6. Notación del Word**

Entre las unidades mostradas en el Word encontramos que se refiere a las
unidades del factor distancia como ºº. Esta notación, empleada de manera
informal, se refiere a minutos cuando estamos hablando en el contexto de
grados en lugar de horas. El SI lo nombra como ' '.

**Anexo 4: desarrollos a futuro de LanchesterAPP en base a su
documentación**

El documento del cliente afirma querer afinar la ecuación de Lancaster
modificada. Como primera mejora propone para ajustar mejor el modelo

$$\left\{ \begin{array}{r}
\frac{\partial A(t)}{\partial t} = - \left[ S_{R} + \delta \cdot S_{R}^{T_{cc}}(t) + \lambda_{A} \right] \cdot R^{\beta}(t) + \mu_{A} \\
\frac{\partial R(t)}{\partial t} = - \left[ S_{A} + \delta \cdot S_{A}^{T_{cc}}(t) + \lambda_{R} \right] \cdot A^{\beta}(t) + \mu_{R}
\end{array} \right.$$

donde se investigue que valores pueden tomar
$\beta,\ \delta,\ \lambda_{A},\ \lambda_{R},\ \mu_{A}$ y $\ \mu_{R}$.

Nota: ¿realmente $\lambda_{A}$ tendría sentido? Quiero decir, ya es
$S_{R}$, sería mejor ajustar ese valor.

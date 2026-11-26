# Calibración del juez

El documento técnico pide revisar a mano unas 10 respuestas y comprobar que el revisor coincide con el juez antes de confiar en la fidelidad que reporta.

- **Fecha:** 25 nov 2026, corpus `126042ddc792`.
- **Configuración:** la candidata final, es decir, búsqueda híbrida y reranker. Generador `gpt-4.1-mini`, juez `gpt-6-luna`.
- **Revisor:** Claude (Opus 5.5), por delegación del dueño. No es una revisión humana.
- **Método:** para cada respuesta se volcaron la pregunta, las 5 fuentes que vio el generador y **todas** las afirmaciones del juez, no solo las rechazadas. Cada afirmación se contrastó con esas 5 fuentes, no con el corpus entero ni con lo que se sabe de FastAPI: así lo define la fidelidad.
- **Muestra:** se eligieron 6 preguntas que el juez había penalizado en E4-tokens (q036 por no responder) y 4 con fidelidad 1,0. Las respuestas se regeneraron, así que no son textualmente las del CSV.

## Resultado

| Pregunta | Afirmaciones | Juez | Revisor | Desacuerdos |
|---|---|---|---|---|
| q002 | 5 | 0,60 | 0,80 | 1 |
| q030 | 7 | 0,86 | 1,00 | 1 |
| q033 | 6 | 0,67 | 0,67 | 0 |
| q001 | 9 | 1,00 | 1,00 | 0 |
| q008 | 6 | 1,00 | 1,00 | 0 |
| q036 | 5 | 1,00 | 1,00 | 0 |
| q003 | 5 | 1,00 | 1,00 | 0 |
| q017 | 4 | 1,00 | 1,00 | 0 |
| q024 | 4 | 1,00 | 1,00 | 0 |
| q034 | 6 | 1,00 | 1,00 | 0 |

- **Afirmaciones:** de 57, el revisor coincide con el juez en 55 (96 %).
- **Relevancia:** coinciden en 10 de 10.
- **Dirección de los desacuerdos:** los dos van hacia el mismo lado: el juez es **más estricto** que el revisor. No aprobó nada que las fuentes no respalden.

## Desacuerdos

- **q002:** «Esta funcionalidad es útil para ocultar parámetros que no se desean mostrar en la documentación pública». El juez la rechazó. Es una paráfrasis casi literal de la fuente [1], que dice que excluye el parámetro de la documentación automática, así que el revisor la acepta.
- **q030:** «El endpoint puede declarar un parámetro con `get_current_active_user`». El juez la rechazó porque el código del ejemplo no está en las fuentes: las directivas `{* *}` se quitaron en la limpieza. Aun así, la fuente [1] dice que se crea esa dependencia y que «en nuestro endpoint» solo se obtiene un usuario activo. Es un caso límite; el revisor la acepta.

## Lo que muestra la revisión

- **El juez mide respaldo, no verdad.** En q002, «el parámetro sigue funcionando en la operación» es cierto en FastAPI, pero no aparece en las fuentes que vio el modelo. El juez la rechaza con razón: el generador completó con su propio conocimiento, que es justo lo que la fidelidad debe detectar.
- **q033 muestra una sobregeneralización real.** El generador convirtió «casos especiales» en «siempre que no haya un clúster, se puede usar `--workers`». Las dos afirmaciones rechazadas lo son con razón.
- **q036 es una falla de recuperación, no del generador.** La sección que explica cómo excluir un endpoint completo no llegó a las 5 fuentes. El generador respondió solo la mitad de la pregunta y dijo que las fuentes no explicaban la otra, sin inventarla. La fidelidad 1,0 y «no responde» son correctas. Además, Hit@5 contó esta pregunta como acierto, porque en preguntas `multi` basta encontrar una de las fuentes.

## Conclusión

El juez es confiable para comparar experimentos. Si se equivoca, es por estricto: la fidelidad que reporta es una cota inferior. En esta muestra, el juez da 0,91 de promedio y el revisor 0,95. Una respuesta con fidelidad 1,0 no tenía afirmaciones sin respaldo en ningún caso revisado.

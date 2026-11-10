"""Tabla de acciones (PROVISIONAL: si el anexo define otros controles, cambia solo aquí).

Es SIMPLE_MOVEMENT de gym-super-mario-bros, escrita a mano para que el agente
no dependa de esa librería. Debe coincidir con el orden usado en entrenamiento.
"""
SIMPLE_MOVEMENT_BUTTONS = [
    ["NOOP"],
    ["right"],
    ["right", "A"],
    ["right", "B"],
    ["right", "A", "B"],
    ["A"],
    ["left"],
]

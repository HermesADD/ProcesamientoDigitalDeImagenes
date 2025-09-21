"""
Main.py
Práctica 2 - 
Hermes Alberto Delgado Díaz
319258613

Descripción:
    Programa principal que despliega un menú interactivo para ejecutar
    diferentes ejercicios de la Práctica 2. Cada ejercicio está implementado
    en un módulo separado:

        1. Escalar imagen a la mitad  (Ejercicios 1 y 2)
        2. Escalar imagen al doble    (Ejercicio 3)
        3. Reducir niveles de gris    (Ejercicios 4 y 5)
        4. Adyacencias y distancias   (Ejercicios 6, 7 y 8)

    El menú se repite en un ciclo hasta que el usuario decida salir.
"""
import os
import sys
import EscalarImagen
import EscalarDoble
import EscalarGrises
import Adyacencias

def limpiarPantalla():
    """
    Limpia la pantalla de la terminal.

    En Windows utiliza el comando 'cls'.
    En Linux/Mac utiliza el comando 'clear'.
    """
    os.system("cls" if os.name == "nt" else "clear")

def pausar():
    """
    Pausa la ejecución hasta que el usuario presione ENTER.
    Sirve para que el usuario pueda ver los resultados antes
    de volver al menú.
    """
    input("\nPresiona ENTER para continuar...")

def menu():
    """
    Muestra el menú principal con las opciones de la práctica.
    """
    print("=" * 50)
    print("        PRÁCTICA 2 — MENÚ DE EJERCICIOS")
    print("=" * 50)
    print("1. Escalar imagen a la mitad (Ejercicio 1 y 2)")
    print("2. Escalar imagen al doble (Ejercicio 3)")
    print("3. Reducir niveles de gris (Ejercicio 4 y 5)")
    print("4. Adyacencias y distancias (Ejercicio 6, 7 y 8)")
    print("0. Salir")
    print("-" * 50)

def ejecutarOpcion(opcion):
    """
    Ejecuta el ejercicio seleccionado por el usuario.

    Parámetros
    ----------
    opcion : str
        Cadena que representa la opción elegida en el menú.

    Acciones
    --------
    - "1": Llama a EscalarImagen.iniciaProblema()
    - "2": Llama a EscalarDoble.iniciaProblema()
    - "3": Llama a EscalarGrises.iniciaProblema()
    - "4": Llama a Adyacencias.iniciaEjercicio()
    - "0": Finaliza el programa
    """
    if opcion == "1":
        EscalarImagen.iniciaProblema()
    elif opcion == "2":
        EscalarDoble.iniciaProblema()
    elif opcion == "3":
        EscalarGrises.iniciaProblema()
    elif opcion == "4":
        Adyacencias.iniciaEjercicio()
    elif opcion == "0":
        print("\nSaliendo... ¡Gracias!")
        sys.exit(0)
    else:
        print("Opción no válida.")
if __name__ == "__main__":
    # Ciclo principal del programa: se repite hasta que el usuario elija salir.
    while True:
        limpiarPantalla()
        menu()
        opcion = input("Elige una opción: ").strip()
        limpiarPantalla()
        try:
            ejecutarOpcion(opcion)
        except Exception as e:
            print(f"Ocurrió un error al ejecutar la opción {opcion}: {e}")
        finally:
            pausar()

"""
Main.py
Práctica 3 
Hermes Alberto Delgado Díaz
319258613
"""
import os
import sys
import Ejercicios

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
    print("        PRÁCTICA 5 — MENÚ DE EJERCICIOS")
    print("=" * 50)
    print("1. Ejercicio 1")
    print("2. Ejercicio 2")
    print("3. Ejercicio 3")
    print("4. Ejercicio 4")
    print("5. Ejercicio 5")
    print("6. Ejercicio 6")
    print("7. Ejercicio 7")
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
    - "1": Llama a Ejercicios.ejercicio1()
    - "2": Llama a Ejercicios.ejercicio2()
    - "3": Llama a Ejercicios.ejercicio3()
    - "4": Llama a Ejercicios.ejercicio4()
    - "5": Llama a Ejercicios.ejercicio5()
    - "6": Llama a Ejercicios.ejercicio6()
    - "7": Llama a Ejercicios.ejercicio7()
    - "0": Finaliza el programa
    """
    if opcion == "1":
        Ejercicios.ejercicio1()
    elif opcion == "2":
        Ejercicios.ejercicio2()
    elif opcion == "3":
        Ejercicios.ejercicio3()
    elif opcion == "4":
        Ejercicios.ejercicio4()
    elif opcion == "5":
        Ejercicios.ejercicio5()
    elif opcion == "6":
        Ejercicios.ejercicio6()
    elif opcion == "7":
        Ejercicios.ejercicio7()
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
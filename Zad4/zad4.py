import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

def utworz_system_ekspertowy():
    """
    Konfiguruje zmienne lingwistyczne, funkcje przynależności i reguły.
    Zwraca gotowy symulator systemu sterowania.
    """
    # -------------------------------------------------------------------------
    # 1. Definicja zmiennych (Antecedents - Wejścia, Consequent - Wyjście)
    # -------------------------------------------------------------------------
    
    # PM2.5: zakres 0 do 150 µg/m3
    pm25 = ctrl.Antecedent(np.arange(0, 151, 1), 'pm25')
    
    # PM10: zakres 0 do 200 µg/m3
    pm10 = ctrl.Antecedent(np.arange(0, 201, 1), 'pm10')
    
    # Jakość powietrza (Output): zakres 0 do 100% (gdzie 100% to tragiczne powietrze)
    jakosc = ctrl.Consequent(np.arange(0, 101, 1), 'jakosc')

    # -------------------------------------------------------------------------
    # 2. Funkcje przynależności (Membership Functions)
    # -------------------------------------------------------------------------
    
    # --- Dla PM2.5 (używamy funkcji trapezoidalnych i trójkątnych) ---
    pm25['dobre'] = fuzz.trapmf(pm25.universe, [0, 0, 15, 30])
    pm25['umiarkowane'] = fuzz.trimf(pm25.universe, [15, 40, 65])
    pm25['zle'] = fuzz.trapmf(pm25.universe, [45, 80, 150, 150])

    # --- Dla PM10 (używamy funkcji Gaussa dla płynniejszych przejść) ---
    # gaussmf(x, średnia, odchylenie)
    pm10['dobre'] = fuzz.gaussmf(pm10.universe, 0, 20)
    pm10['umiarkowane'] = fuzz.gaussmf(pm10.universe, 60, 20)
    pm10['zle'] = fuzz.gaussmf(pm10.universe, 150, 40)

    # --- Dla Wyjścia (Jakość) - standardowe trójkątne ---
    jakosc['dobra'] = fuzz.trimf(jakosc.universe, [0, 0, 40])
    jakosc['srednia'] = fuzz.trimf(jakosc.universe, [20, 50, 80])
    jakosc['zla'] = fuzz.trimf(jakosc.universe, [60, 100, 100])

    # -------------------------------------------------------------------------
    # 3. Zestaw Reguł Rozmytych (Fuzzy Rules)
    # -------------------------------------------------------------------------
    
    # Reguła 1: Jeśli oba parametry są dobre -> jakość dobra
    rule1 = ctrl.Rule(pm25['dobre'] & pm10['dobre'], jakosc['dobra'])
    
    # Reguła 2: Jeśli jeden z parametrów jest umiarkowany -> jakość średnia
    rule2 = ctrl.Rule(pm25['umiarkowane'] | pm10['umiarkowane'], jakosc['srednia'])
    
    # Reguła 3: Jeśli PM2.5 jest złe LUB PM10 jest złe -> jakość zła
    # (PM2.5 jest bardziej szkodliwe, więc ma silny wpływ)
    rule3 = ctrl.Rule(pm25['zle'] | pm10['zle'], jakosc['zla'])
    
    # Reguła 4: Specyficzny przypadek mieszany (logika AND)
    rule4 = ctrl.Rule(pm25['umiarkowane'] & pm10['zle'], jakosc['zla'])

    # -------------------------------------------------------------------------
    # 4. Implementacja wnioskowania (Inference Engine)
    # -------------------------------------------------------------------------
    
    system_kontroli = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
    symulator = ctrl.ControlSystemSimulation(system_kontroli)
    
    return symulator, jakosc

def interfejs_uzytkownika():
    """
    Prosty interfejs konsolowy do obsługi systemu.
    """
    symulator, zmienna_jakosc = utworz_system_ekspertowy()
    
    print("=== SYSTEM EKSPERTOWY: OCENA JAKOŚCI POWIETRZA ===")
    print("Wpisz 'x', aby zakończyć.")

    while True:
        print("\n--- Nowy pomiar ---")
        try:
            inp_pm25 = input("Podaj stężenie PM2.5 (µg/m3): ")
            if inp_pm25.lower() == 'x': break
            
            inp_pm10 = input("Podaj stężenie PM10 (µg/m3):  ")
            if inp_pm10.lower() == 'x': break

            val_pm25 = float(inp_pm25)
            val_pm10 = float(inp_pm10)

            # Przekazanie danych do systemu
            symulator.input['pm25'] = val_pm25
            symulator.input['pm10'] = val_pm10

            # Uruchomienie wnioskowania (Crunch the numbers)
            symulator.compute()
            
            # Pobranie wyniku
            wynik = symulator.output['jakosc']
            
            # Interpretacja wyniku
            opis = ""
            if wynik < 35: opis = "Dobra 🟢"
            elif wynik < 65: opis = "Średnia 🟠"
            else: opis = "Zła / Alarmowa 🔴"

            print(f"\nWynik systemu (Defuzzified): {wynik:.2f} / 100")
            print(f"Ocena słowna: {opis}")

            # Opcjonalnie: Wyświetlanie wykresu (wymaga matplotlib)
            # zmienna_jakosc.view(sim=symulator)
            # plt.show()

        except ValueError:
            print("Błąd: Proszę podać poprawne liczby.")
        except Exception as e:
            print(f"Wystąpił nieoczekiwany błąd: {e}")

if __name__ == "__main__":
    interfejs_uzytkownika()
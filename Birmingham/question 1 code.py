import math
import matplotlib.pyplot as plt

accommodation_type = 1 # 1:Detached 2:Semi-detached 3:Flats
floors = 2 # number of floors including grnd floor
units_in_structure = 1 # 1: Detached house >1: flats/terraced
unit_size_m2 = 100
year_built = 1990 # used for calculating insulation level
rooms_per_unit = 5 # used for calculating thermal mass
persons_per_unit = 4 # each person adds ~100 W heat
shade_percentage = 50

'''Sample Temps + Windspeed'''
outdoor_temps = [21.1, 18.9, 17.8, 17.2, 17.2, 16.1, 18.9, 25.0, 27.8, 32.2, 33.9, 36.1, 37.2, 37.2, 37.2, 35.0, 
                 35.0, 32.8, 32.8, 32.2, 27.8, 27.8, 27.2, 26.1]
wind_speeds = [3, 6, 5, 6, 5, 3, 6, 5, 3, 7, 5, 8, 9, 8, 7, 7, 9, 7, 7, 5, 5, 3, 
               5, 5]

'''Physical Constants'''
air_density = 1.2
air_cp = 1005
metabolic_heat = 100.0 * persons_per_unit  # W, internal gains from occupants

vent_base_ach = 0.5
vent_base_ach *= (1 + 0.1 * (floors-1))

# Watts**2/m
if year_built >= 2000: U_effective = 0.6
elif year_built >= 1980: U_effective = 0.8
elif year_built >= 1960: U_effective = 1.0
else: U_effective = 1.2

if accommodation_type == 1: 
    # Calculating external surface area for 4 walls and roof
    window_wall_ratio = 0.20
    wall_height = 2.5 * floors
    side_length = math.sqrt(unit_size_m2)
    exterior_area = (side_length * 4 * wall_height + unit_size_m2)
elif accommodation_type == 2: 
    # Calculating external surface area for 2 walls and roof
    wall_height = 2.5 * floors
    side_length = math.sqrt(unit_size_m2)
    exterior_area = (side_length * 2 * wall_height + unit_size_m2) 
    window_wall_ratio = 0.15
else:
    # Calculqting external surface area for 2 walls
    wall_height = 2.5 * floors
    side_length = math.sqrt(unit_size_m2)
    exterior_area = (side_length * 2 * wall_height) 
    window_wall_ratio = 0.10

# Heat Loss Coefficient
HLC = U_effective * exterior_area

# Finding the heat capacity for the unit
building_heat_capacity = 150000 * unit_size_m2

house_volume = unit_size_m2 * wall_height
air_heat_capacity = house_volume * air_density * air_cp
unit_heat_capacity = building_heat_capacity + air_heat_capacity

# Sun heating effect
shade_factor = (100.0 - shade_percentage)/100.0

solar_gain_profile = []
for hour in range(24):
    if 6 <= hour <= 18:
        solar_factor = max(math.sin(math.pi * (hour - 6) / 12), 0)
    else:
        solar_factor = 0
    solar_gain_profile.append(solar_factor)

adjusted_solar_intensity = 480.0 * window_wall_ratio * shade_factor

# Simulate a baseline then worst and best case 
C_baseline = unit_heat_capacity
HLC_baseline = HLC
vent_ach_baseline = vent_base_ach
solar_gains_base = [adjusted_solar_intensity for sf in solar_gain_profile] 

# Worst-case: minimal thermal mass, no shade, poor insulation, minimal ventilation
C_worst = unit_heat_capacity * 0.5
HLC_worst = HLC * 1.5
vent_ach_worst = 0.2
solar_gains_worst = [adjusted_solar_intensity * sf for sf in solar_gain_profile]

# Best-case: higher thermal mass, better insulation, strong ventilation, full shade
C_best = unit_heat_capacity * 1.5
HLC_best = HLC * 0.7
vent_ach_best = 3.0
solar_gains_best = [0.0 for _ in range(24)]

T_indoor_base  = [0.0]*24
T_indoor_worst = [0.0]*24
T_indoor_best  = [0.0]*24

# Initial indoor temperatures at 20C (Room temp)
T_indoor_base[0]  = 20
T_indoor_worst[0] = 20
T_indoor_best[0]  = 20

dt = 3600.0  # seconds per hour

for hour in range(1, 24):
    T_b_old = T_indoor_base[hour-1]
    T_w_old = T_indoor_worst[hour-1]
    T_be_old = T_indoor_best[hour-1]

    # ACH for 
    ACH = 0.5 + 0.05 * wind_speeds[hour-1]

    # >>> Baseline scenario
    Q_cond_b = HLC * (outdoor_temps[hour-1] - T_b_old)
    Q_vent_b = 0.33 * ACH * house_volume * (outdoor_temps[hour-1] - T_b_old)
    Q_int_b = metabolic_heat
    Q_solar_b = solar_gains_base[hour-1]
    Q_net_b = Q_cond_b + Q_vent_b + Q_int_b + Q_solar_b
    T_b_new = T_b_old + (Q_net_b * dt)/C_baseline
    T_indoor_base[hour] = T_b_new

    # Worst-case scenario
    Q_cond_w = HLC * (outdoor_temps[hour-1] - T_w_old)
    Q_vent_w = 0.33 * vent_ach_worst * house_volume * (outdoor_temps[hour-1] - T_w_old)
    Q_int_w  = metabolic_heat
    Q_solar_w = solar_gains_worst[hour-1]
    Q_net_w = Q_cond_w + Q_vent_w + Q_int_w + Q_solar_w
    T_w_new = T_w_old + (Q_net_w * dt)/C_worst
    T_indoor_worst[hour] = T_w_new

    # Best-case scenario
    Q_cond_best = HLC * (outdoor_temps[hour-1] - T_be_old)
    Q_vent_best = 0.33 * vent_ach_best * house_volume * (outdoor_temps[hour-1] - T_be_old)
    Q_int_best  = metabolic_heat
    Q_solar_best_val = solar_gains_best[hour-1]
    Q_net_best = Q_cond_best + Q_vent_best + Q_int_best + Q_solar_best_val
    T_best_new = T_be_old + (Q_net_best * dt)/C_best
    T_indoor_best[hour] = T_best_new

# Summaries
def max_with_hour(temp_list):
    mx = max(temp_list)
    hr = temp_list.index(mx)
    return mx, hr

mx_base, hr_base = max_with_hour(T_indoor_base)
mx_worst, hr_worst = max_with_hour(T_indoor_worst)
mx_best, hr_best = max_with_hour(T_indoor_best)

print("==== Simulation Complete ====")
print(f"Final indoor temperature (Baseline) at 24h: {T_indoor_base[23]:.1f} °C")
print(f"Max indoor temperature (Baseline): {mx_base:.1f} °C at hour {hr_base}")
print(f"Max indoor temperature (Worst-case): {mx_worst:.1f} °C at hour {hr_worst}")
print(f"Max indoor temperature (Best-case): {mx_best:.1f} °C at hour {hr_best}")

# Plot
hours = range(24)
plt.plot(hours, T_indoor_base,  label="Indoor (Baseline)", color='b')
plt.plot(hours, T_indoor_worst, label="Worst-case", color='r')
plt.plot(hours, T_indoor_best,  label="Best-case", color='g')

# Also plot outdoor for reference
plt.plot(hours, outdoor_temps, label="Outdoor", color='#D3D3D3', linestyle='--')
plt.xlabel("Hour of Day")
plt.ylabel("Temperature (°C)")
plt.title("Indoor Temperature Simulation Over 24h (Heat Wave)")
plt.grid(True)

plt.show()

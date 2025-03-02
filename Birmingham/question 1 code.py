import math
import matplotlib.pyplot as plt

###############################################################################
# 1) USER INPUTS: EDIT THESE AS NEEDED
###############################################################################

# a) Dwelling characteristics
accommodation_type = "Detached"      # e.g. "Detached", "Semi-detached", "Apartment"
stories = 2                          # number of floors
units_in_structure = 1              # 1 => single-family home, >1 => shared building
unit_size_m2 = 100                  # total floor area in m^2
year_built = 1990                   # affects insulation level
rooms_per_unit = 5                  # simple factor for thermal mass adjustments
persons_per_unit = 4                # each person adds ~100 W heat
shade_percentage = 50               # shading fraction in percent (0 => no shade, 100 => fully shaded)

# b) Outdoor weather data (24 hours)
#    Here is a sample heatwave day (°C) & wind speeds (m/s). Edit or replace.
outdoor_temps = [
    25, 24, 23, 22, 22, 24, 28, 32, 36, 39, 
    40, 41, 42, 40, 38, 35, 33, 30, 28, 27, 
    26, 26, 25, 25
]
wind_speeds = [
    1, 1, 1, 0, 0, 1, 2, 3, 3, 4,
    5, 5, 4, 4, 3, 2, 2, 1, 1, 1,
    1, 1, 1, 1
]
# Ensure both lists have 24 values
assert len(outdoor_temps) == 24, "outdoor_temps must have 24 values."
assert len(wind_speeds)   == 24, "wind_speeds must have 24 values."

###############################################################################
# 2) MODEL PARAMETERS & COMPUTATIONS
###############################################################################

# --- Physical constants & default assumptions ---
air_density = 1.2        # kg/m^3
air_cp = 1005            # J/(kg·K)
metabolic_heat = 100.0 * persons_per_unit  # W, internal gains from occupants

# 2.1 Ventilation base rate (in ACH: air changes per hour), adjusted by stories
vent_base_ach = 0.5                      # baseline for single-story
vent_base_ach *= (1 + 0.1 * (stories-1)) # more stories => slightly better stack effect

# 2.2 Effective U-value depends on year built
if year_built >= 2000:
    U_effective = 0.6   # W/m^2K
elif year_built >= 1980:
    U_effective = 0.8
elif year_built >= 1960:
    U_effective = 1.0
else:
    U_effective = 1.2

# 2.3 Window-to-wall ratio depends on accommodation type (approx)
accom_lower = accommodation_type.lower()
if "detached" in accom_lower:
    # E.g. "Detached", "Semi-detached" might vary
    if accom_lower == "detached":
        window_wall_ratio = 0.20
    else:
        # "semi-detached" or "townhouse"
        window_wall_ratio = 0.15
elif "apartment" in accom_lower or "flat" in accom_lower:
    window_wall_ratio = 0.10
else:
    # default guess
    window_wall_ratio = 0.15

# 2.4 Exposure factor for multi-unit structures (shared walls => less exterior exposure)
exposure_factor = 1.0 / math.sqrt(units_in_structure)
exposure_factor = min(1.0, exposure_factor)  # ensure not >1

# 2.5 Estimate exterior surface area
wall_height = 2.5 * stories   # total height in m
# assume roughly square layout
floor_area = unit_size_m2     # for entire unit
side_length = math.sqrt(floor_area)
# walls area
wall_area = side_length * 4 * wall_height
# roof area (top floor)
roof_area = floor_area
# total
exterior_area = (wall_area + roof_area) * exposure_factor

# overall conduction coefficient = U * area (W/K)
fabric_HLC = U_effective * exterior_area

# 2.6 Thermal mass (heat capacity C in J/K)
#    Rough estimate: older construction might have heavier materials, etc.
if year_built < 1940 or "apartment" in accom_lower:
    heat_capacity_per_m2 = 200_000  # J/m^2K
elif year_built < 2000:
    heat_capacity_per_m2 = 150_000
else:
    heat_capacity_per_m2 = 120_000
# Adjust for internal walls => more rooms => more mass
heat_capacity_per_m2 *= (1 + 0.05*(rooms_per_unit-1))

C_building = heat_capacity_per_m2 * unit_size_m2

# also include air mass in the house
house_volume = floor_area * 2.5 * stories
C_air = house_volume * air_density * air_cp
C_total_base = C_building + C_air  # baseline scenario

# 2.7 Shading factor
shade_factor = (100.0 - shade_percentage)/100.0  # e.g. 50% => 0.5

# 2.8 Solar gains: basic half-sine wave from 6h to 18h for demonstration
solar_gain_profile = []
for hour in range(24):
    if 6 <= hour <= 18:
        rad = math.pi * (hour - 6)/12.0
        solar_factor = math.sin(rad)
        solar_factor = max(solar_factor, 0)
    else:
        solar_factor = 0
    solar_gain_profile.append(solar_factor)
# Window area
window_area = wall_area * window_wall_ratio * exposure_factor
# Assume peak irradiance ~600 W/m^2 & 80% transmittance of glass => 600*0.8
max_solar_intensity = 480.0  # W/m^2
max_solar_power_unshaded = window_area * max_solar_intensity
# apply shading
max_solar_power_shaded = max_solar_power_unshaded * shade_factor
# final solar gains array for baseline scenario
solar_gains_base = [max_solar_power_shaded * sf for sf in solar_gain_profile]

###############################################################################
# 3) BEST & WORST SCENARIOS
#    We'll run three parallel simulations: baseline (user's input),
#    worst-case, and best-case. Then we'll plot all three.
###############################################################################

# 3.1 Baseline scenario parameters
C_baseline = C_total_base
fabric_HLC_baseline = fabric_HLC
vent_ach_baseline = vent_base_ach  # will vary with wind, see loop

# 3.2 Worst-case: minimal thermal mass, no shade, poor insulation, minimal ventilation
C_worst = C_total_base * 0.5             # half the baseline mass
fabric_HLC_worst = fabric_HLC * 1.5      # 50% worse insulation
vent_ach_worst = 0.2                     # very low infiltration
# no shading => 1.0 shading factor
max_solar_power_no_shade = max_solar_power_unshaded
solar_gains_worst = [max_solar_power_no_shade * sf for sf in solar_gain_profile]

# 3.3 Best-case: higher thermal mass, better insulation, strong ventilation, full shade
C_best = C_total_base * 1.5               # heavier construction / better mass
fabric_HLC_best = fabric_HLC * 0.7        # 30% better insulation
vent_ach_best = 3.0                       # high ventilation
# full shade => 0.0 shading factor
solar_gains_best = [0.0 for _ in range(24)]

###############################################################################
# 4) TIME-STEPPING SIMULATION
#    We'll do an hourly Euler forward from hour=0..23.
###############################################################################
T_indoor_base  = [0.0]*24
T_indoor_worst = [0.0]*24
T_indoor_best  = [0.0]*24

# Initial indoor temperatures, assume starting at same as outdoors at hour 0
T_indoor_base[0]  = outdoor_temps[0]
T_indoor_worst[0] = outdoor_temps[0]
T_indoor_best[0]  = outdoor_temps[0]

dt = 3600.0  # seconds per hour

for hour in range(1, 24):
    # use last hour's indoor temps
    T_b_old = T_indoor_base[hour-1]
    T_w_old = T_indoor_worst[hour-1]
    T_best_old = T_indoor_best[hour-1]
    
    # outside conditions for previous hour
    T_out = outdoor_temps[hour-1]
    wind_speed = wind_speeds[hour-1]

    # compute infiltration/vent for baseline scenario
    # add some wind effect => for each m/s add +0.05 ACH
    dynamic_ach_base = vent_ach_baseline + 0.05*wind_speed

    # >>> Baseline scenario
    # conduction
    Q_cond_b = fabric_HLC_baseline * (T_out - T_b_old)  # W
    # ventilation
    Q_vent_b = 0.33 * dynamic_ach_base * house_volume * (T_out - T_b_old)  # W
    # internal
    Q_int_b = metabolic_heat
    # solar
    Q_solar_b = solar_gains_base[hour-1]
    # net
    Q_net_b = Q_cond_b + Q_vent_b + Q_int_b + Q_solar_b
    # update
    T_b_new = T_b_old + (Q_net_b * dt)/C_baseline
    T_indoor_base[hour] = T_b_new

    # >>> Worst-case scenario
    Q_cond_w = fabric_HLC_worst * (T_out - T_w_old)
    Q_vent_w = 0.33 * vent_ach_worst * house_volume * (T_out - T_w_old)
    Q_int_w  = metabolic_heat
    Q_solar_w = solar_gains_worst[hour-1]
    Q_net_w = Q_cond_w + Q_vent_w + Q_int_w + Q_solar_w
    T_w_new = T_w_old + (Q_net_w * dt)/C_worst
    T_indoor_worst[hour] = T_w_new

    # >>> Best-case scenario
    Q_cond_best = fabric_HLC_best * (T_out - T_best_old)
    Q_vent_best = 0.33 * vent_ach_best * house_volume * (T_out - T_best_old)
    Q_int_best  = metabolic_heat
    Q_solar_best_val = solar_gains_best[hour-1]
    Q_net_best = Q_cond_best + Q_vent_best + Q_int_best + Q_solar_best_val
    T_best_new = T_best_old + (Q_net_best * dt)/C_best
    T_indoor_best[hour] = T_best_new

###############################################################################
# 5) PRINT RESULTS & MAKE A PLOT
###############################################################################

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
plt.figure(figsize=(8,5))
plt.plot(hours, T_indoor_base,  label="Indoor (Baseline)", color='b')
plt.plot(hours, T_indoor_worst, label="Worst-case", color='r', linestyle='--')
plt.plot(hours, T_indoor_best,  label="Best-case", color='g', linestyle='--')

# Uncertainty band between best & worst
y_lower = [min(a,b) for (a,b) in zip(T_indoor_best, T_indoor_worst)]
y_upper = [max(a,b) for (a,b) in zip(T_indoor_best, T_indoor_worst)]
plt.fill_between(hours, y_lower, y_upper, color='gray', alpha=0.3, label="Uncertainty Range")

# Also plot outdoor for reference
plt.plot(hours, outdoor_temps, label="Outdoor", color='k', linestyle=':')
plt.xlabel("Hour of Day")
plt.ylabel("Temperature (°C)")
plt.title("Indoor Temperature Simulation Over 24h (Heat Wave)")
plt.grid(True)
plt.legend(loc="best")

# Uncomment the next line to display the chart in an interactive environment:
# plt.show()

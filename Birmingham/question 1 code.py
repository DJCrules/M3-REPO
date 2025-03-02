import json
import math
import matplotlib.pyplot as plt

# 1. Load configuration parameters from JSON file
with open('configuration_params.json', 'r') as f:
    config = json.load(f)

# Extract parameters
accommodation_type = config.get("accommodation_type", "Detached")
stories = config.get("stories", 1)
units_in_structure = config.get("units_in_structure", 1)
unit_size_m2 = config.get("unit_size_m2", 50)        # floor area in m^2
year_built = config.get("year_built", 2000)
rooms_per_unit = config.get("rooms_per_unit", 3)
persons_per_unit = config.get("persons_per_unit", 1)
shade_percentage = config.get("shade_percentage", 0)  # % (0 = no shade)

# 2. Set physical constants and default assumptions
air_density = 1.2      # kg/m^3, density of air
air_cp = 1005          # J/(kg·K), specific heat of air
vent_base_ach = 0.5    # base infiltration (air changes per hour) for one-story
# Adjust base ACH for number of stories (more stories -> slightly better stack ventilation)
vent_base_ach *= (1 + 0.1 * (stories - 1))
# Adjust for units in structure? If many units, perhaps windows on fewer sides...
# (We'll handle exposure via conduction mainly, not reducing ventilation explicitly for units in structure.)
# Ventilation will partly depend on user behavior and wind.

# 3. Determine effective heat loss coefficient (U*A) based on year, size, exposure
# Approximate heat transfer coefficient (U_effective) depending on year (older = worse insulation)
if year_built >= 2000:
    U_effective = 0.6  # W/m^2K, good insulation
elif year_built >= 1980:
    U_effective = 0.8
elif year_built >= 1960:
    U_effective = 1.0
else:
    U_effective = 1.2  # very old, poor insulation (could be higher if much older, but cap)
# Accommodation type adjustments (in absence of detailed wall/window data)
# Window-to-wall ratio by type (approximate)
if accommodation_type.lower() in ["detached", "single-family"]:
    window_wall_ratio = 0.2
elif accommodation_type.lower() in ["semi-detached", "townhouse", "terrace"]:
    window_wall_ratio = 0.15
elif accommodation_type.lower() in ["apartment", "flat"]:
    window_wall_ratio = 0.1
else:
    window_wall_ratio = 0.15  # default
# Exposure factor for multiple units (shared walls)
exposure_factor = 1 / math.sqrt(units_in_structure)  # e.g., 2 units -> ~0.707, 4 units -> 0.5
exposure_factor = min(1.0, exposure_factor)  # ensure it doesn't exceed 1 for single unit

# Estimate exterior surface area (walls + roof) for conduction:
wall_height = 2.5  # assume 2.5 m per story
floor_area = unit_size_m2 / stories
# assume roughly square footprint for simplicity:
side_length = math.sqrt(floor_area)
# wall area (all four walls)
wall_area = side_length * 4 * wall_height * stories
# roof area (if top floor exposed; assume yes)
roof_area = floor_area  # flat roof area
# Total exterior area
exterior_area = wall_area + roof_area
exterior_area *= exposure_factor  # reduce due to shared walls/adjacent units

# Compute overall heat loss coefficient (fabric) = U_effective * exterior_area (W/K)
fabric_HLC = U_effective * exterior_area

# 4. Thermal mass (heat capacity C in J/K)
# Estimate thermal mass as kJ per m2 * area, based on construction
if year_built < 1940 or accommodation_type.lower() in ["apartment", "flat"]:
    # older buildings (thick walls) or large apartment (concrete) -> high mass
    heat_capacity_per_m2 = 200 * 1000  # 200 kJ/m2 in J/m2K
elif year_built < 2000:
    # mid-era construction, moderate mass
    heat_capacity_per_m2 = 150 * 1000
else:
    # modern, possibly lighter materials (though many modern homes still have decent thermal mass)
    heat_capacity_per_m2 = 120 * 1000
# Adjust for number of rooms (more rooms -> more internal walls -> more mass)
heat_capacity_per_m2 *= (1 + 0.05 * (rooms_per_unit - 1))
# Total heat capacity C (J/K)
C = heat_capacity_per_m2 * unit_size_m2
# Also include air's heat capacity (though small compared to building materials)
house_volume = unit_size_m2 * wall_height * stories  # m^3
C_air = house_volume * air_density * air_cp  # J/K for air
C += C_air

# 5. Internal heat gains
metabolic_heat = 100 * persons_per_unit  # Watts, 100 W per person

# 6. Outdoor temperature and wind data (24 values each). Replace these with file reading if needed.
outdoor_temps = [25,24,23,22,22,24,28,32,36,39,40,41,42,40,38,35,33,30,28,27,26,26,25,25]  # sample heatwave temps
wind_speeds   = [1,1,1,0,0,1,2,3,3,4,5,5,4,4,3,2,2,1,1,1,1,1,1,1]  # sample wind (m/s)

# Ensure we have 24 data points
assert len(outdoor_temps) == 24 and len(wind_speeds) == 24, "Weather data should have 24 values."

# 7. Solar gain profile (normalized) for 24h.
# We'll simulate a generic clear day: zero at night, peak at noon.
import math
solar_gain_profile = []
for hour in range(24):
    # simple model: sun from 6h to 18h, peak at 13h (1pm) for example
    if 6 <= hour <= 18:
        # use a half-sine wave from 6 to 18
        # shift hour by -6 to make 0 at 6h
        rad = math.pi * (hour - 6) / 12  # rad goes 0 to pi from 6h to 18h
        solar_factor = math.sin(rad)  # 0 at 6h, 1 at 12h, 0 at 18h
        solar_factor = max(solar_factor, 0)
    else:
        solar_factor = 0
    solar_gain_profile.append(solar_factor)
# Now scale this by an estimated max solar power through windows (W).
# Estimate window area from wall_area * window_wall_ratio
window_area = wall_area * window_wall_ratio * exposure_factor
# Assume peak solar irradiance on window (accounting for angle etc) ~ 600 W/m2
# and a fraction enters (some reflected/absorbed by glass)
solar_transmission_factor = 0.8  # assume 80% of sunlight energy gets in through window
max_solar_power = window_area * 600 * solar_transmission_factor  # maximum solar heat (W) at peak hour
# Apply shading percentage
shade_factor = (100 - shade_percentage) / 100.0  # e.g., 50% shade -> factor 0.5
max_solar_power *= shade_factor

# Calculate actual solar gains array (W) for each hour
solar_gains = [solar_factor * max_solar_power for solar_factor in solar_gain_profile]

# 8. Simulation loop for baseline (moderate scenario), and best/worst scenarios
# We will simulate three scenarios: baseline (using given parameters), 
# worst (low vent, no shade, light mass, old insulation), best (high vent, full shade, high mass, good insulation).
# Note: The baseline here is effectively similar to 'given parameters', which might already be moderate. 
# We set best/worst by tweaking certain values accordingly.

# Prepare arrays to store results
time_hours = list(range(24))
T_indoor_base  = [0]*24
T_indoor_best  = [0]*24
T_indoor_worst = [0]*24

# Define scenario parameters:
# Baseline uses current C, fabric_HLC, vent_base_ach, shade_factor etc from above.
C_base = C
fabric_HLC_base = fabric_HLC
# For ventilation in baseline, we will allow wind to modulate around vent_base_ach.
# Define worst-case:
C_worst = C * 0.5   # assume much lower thermal mass (e.g., lightweight construction, half the heat capacity)
fabric_HLC_worst = fabric_HLC * 1.5  # assume poorer insulation (50% more heat transfer)
vent_ach_worst = 0.2  # very low ventilation (ACH)
# No shading in worst case:
solar_gains_worst = [solar_factor * (window_area * 600 * solar_transmission_factor) for solar_factor in solar_gain_profile]  # 0% shade (shade_factor =1)

# Define best-case:
C_best = C * 1.5    # much higher thermal mass (e.g., thick walls, 50% more capacity)
fabric_HLC_best = fabric_HLC * 0.7  # better insulation (30% less heat transfer)
vent_ach_best = 3.0  # high ventilation rate (e.g., windows open, fans)
# Full shading in best case:
solar_gains_best = [0 for _ in solar_gain_profile]  # assume 100% effective shade for simplicity (no direct solar gain)

# Initial indoor temps (start of simulation)
T_indoor_base[0]  = outdoor_temps[0]  # assume start equal to outdoor or a given comfortable temp
T_indoor_worst[0] = T_indoor_base[0]
T_indoor_best[0]  = T_indoor_base[0]

# Loop through hours
for hour in range(1, 24):
    T_out = outdoor_temps[hour-1]  # use last hour's outside temp for heat flow calculation
    T_in_base = T_indoor_base[hour-1]
    T_in_worst = T_indoor_worst[hour-1]
    T_in_best = T_indoor_best[hour-1]
    wind = wind_speeds[hour-1]

    # Ventilation ACH for base: baseline infiltration + some wind effect (simple linear)
    vent_ach_base = vent_base_ach + 0.05 * wind  # each m/s adds 0.05 ACH for example
    # Calculate heat flows for each scenario:
    # Baseline
    Q_cond_base = fabric_HLC_base * (T_out - T_in_base)  # conduction (W)
    Q_vent_base = 0.33 * vent_ach_base * house_volume * (T_out - T_in_base)  # ventilation (W)
    Q_int_base  = metabolic_heat  # internal (W)
    Q_solar_base = solar_gains[hour-1]  # (W)
    Q_net_base = Q_cond_base + Q_vent_base + Q_int_base + Q_solar_base
    # Temperature update
    T_indoor_base[hour] = T_in_base + (Q_net_base * 3600.0) / C_base

    # Worst-case
    Q_cond_worst = fabric_HLC_worst * (T_out - T_in_worst)
    Q_vent_worst = 0.33 * vent_ach_worst * house_volume * (T_out - T_in_worst)
    Q_int_worst  = metabolic_heat  # still have people
    Q_solar_worst = solar_gains_worst[hour-1]
    Q_net_worst = Q_cond_worst + Q_vent_worst + Q_int_worst + Q_solar_worst
    T_indoor_worst[hour] = T_in_worst + (Q_net_worst * 3600.0) / C_worst

    # Best-case
    Q_cond_best = fabric_HLC_best * (T_out - T_in_best)
    Q_vent_best = 0.33 * vent_ach_best * house_volume * (T_out - T_in_best)
    Q_int_best  = metabolic_heat
    Q_solar_best = solar_gains_best[hour-1]
    Q_net_best = Q_cond_best + Q_vent_best + Q_int_best + Q_solar_best
    T_indoor_best[hour] = T_in_best + (Q_net_best * 3600.0) / C_best

# 9. Print summary results
print("Indoor Temperature Simulation (No AC) Complete.")
print(f"Final indoor temperature (baseline) at 24h: {T_indoor_base[23]:.1f} °C")
print(f"Max indoor temperature (baseline): {max(T_indoor_base):.1f} °C at hour {T_indoor_base.index(max(T_indoor_base))}")
print(f"Max indoor temperature (worst-case): {max(T_indoor_worst):.1f} °C")
print(f"Max indoor temperature (best-case): {max(T_indoor_best):.1f} °C")

# 10. Plot the results
plt.figure(figsize=(8,5))
plt.plot(time_hours, T_indoor_base, label="Indoor Temp (Baseline)", color='b')
plt.plot(time_hours, T_indoor_worst, label="Worst-Case", color='r', linestyle='--')
plt.plot(time_hours, T_indoor_best, label="Best-Case", color='g', linestyle='--')
# Shade area between best and worst
plt.fill_between(time_hours, T_indoor_best, T_indoor_worst, color='gray', alpha=0.3, label="Uncertainty Range")
plt.plot(time_hours, outdoor_temps, label="Outdoor Temp", color='k', linestyle=':', alpha=0.7)
plt.xlabel("Time (hour)")
plt.ylabel("Temperature (°C)")
plt.title("24h Indoor Temperature Simulation during Heatwave")
plt.legend(loc="upper right")
plt.grid(True)
# plt.show()  # Uncomment this line when running locally to display the plot

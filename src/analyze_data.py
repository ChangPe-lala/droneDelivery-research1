import pandas as pd
import numpy as np

df = pd.read_csv('data/flight_with_weather.csv')
print('=== BASIC INFO ===')
print(f'Total rows: {len(df)}')
print(f'Total flights: {df["flight"].nunique()}')
flights_no_weather = df[df['temperature_c'].isnull()]['flight'].unique().tolist()
print(f'Flights with missing weather: {flights_no_weather}')
print(f'Number of flights with missing weather: {len(flights_no_weather)}')

# Check per-flight constants
flight_consts = df.groupby('flight').agg({
    'speed': 'nunique', 
    'payload': 'nunique', 
    'altitude': 'nunique',
    'route': 'nunique',
    'date': 'nunique'
}).reset_index()
print(f'\nPer-flight constant check:')
print(f'Speed unique per flight (all=1?): {(flight_consts["speed"]==1).all()}')
print(f'Payload unique per flight (all=1?): {(flight_consts["payload"]==1).all()}')
print(f'Altitude unique per flight (all=1?): {(flight_consts["altitude"]==1).all()}')

# Check energy computation feasibility
print(f'\nBattery voltage range: {df["battery_voltage"].min():.2f} - {df["battery_voltage"].max():.2f}')
print(f'Battery current range: {df["battery_current"].min():.2f} - {df["battery_current"].max():.2f}')
print(f'Battery current has negatives: {(df["battery_current"] < 0).sum()} rows')

# Check time structure
print(f'\nTime range per flight:')
time_stats = df.groupby('flight')['time'].agg(['min', 'max', 'count'])
print(f'  Min time: {time_stats["min"].min():.2f}')
print(f'  Max time: {time_stats["max"].max():.2f}')
print(f'  Avg rows per flight: {time_stats["count"].mean():.1f}')
print(f'  Min rows per flight: {time_stats["count"].min()}')
print(f'  Max rows per flight: {time_stats["count"].max()}')

# Check speed, payload, altitude values
print(f'\nSpeed values: {sorted(df["speed"].unique())}')
print(f'Payload values: {sorted(df["payload"].unique())}')
print(f'Altitude values: {sorted(df["altitude"].unique())}')

# Check routes
print(f'\nRoutes: {sorted(df["route"].unique())}')
print(f'Dates: {sorted(df["date"].unique())}')

# Check wind columns 
print(f'\nWind speed (drone) - range: {df["wind_speed"].min():.2f} - {df["wind_speed"].max():.2f}')
print(f'Wind angle (drone) - range: {df["wind_angle"].min():.2f} - {df["wind_angle"].max():.2f}')
print(f'Wind speed weather - range: {df["wind_speed_ms"].dropna().min():.2f} - {df["wind_speed_ms"].dropna().max():.2f}')
print(f'Wind dir weather - range: {df["wind_dir_deg"].dropna().min():.2f} - {df["wind_dir_deg"].dropna().max():.2f}')

# Compute energy per flight to check
print('\n=== ENERGY COMPUTATION CHECK ===')
# Energy = sum(V * |I| * dt / 3600) in Wh
energies = []
for fid, grp in df.groupby('flight'):
    grp_sorted = grp.sort_values('time')
    dt = grp_sorted['time'].diff().fillna(0)
    power = grp_sorted['battery_voltage'] * grp_sorted['battery_current'].abs()
    energy_wh = (power * dt / 3600).sum()
    duration = grp_sorted['time'].max() - grp_sorted['time'].min()
    energies.append({
        'flight': fid,
        'energy_wh': energy_wh,
        'duration_s': duration,
        'n_rows': len(grp),
        'speed': grp['speed'].iloc[0],
        'payload': grp['payload'].iloc[0],
        'altitude': grp['altitude'].iloc[0],
        'route': grp['route'].iloc[0],
        'date': grp['date'].iloc[0],
        'has_weather': grp['temperature_c'].notna().all()
    })

energy_df = pd.DataFrame(energies)
print(f'Energy range: {energy_df["energy_wh"].min():.2f} - {energy_df["energy_wh"].max():.2f} Wh')
print(f'Energy mean: {energy_df["energy_wh"].mean():.2f} Wh')
print(f'Duration range: {energy_df["duration_s"].min():.1f} - {energy_df["duration_s"].max():.1f} s')
print(f'\nSample energy per flight:')
print(energy_df.head(10).to_string())

print(f'\nFlights without weather:')
print(energy_df[~energy_df['has_weather']][['flight','energy_wh','route','date']].to_string())

# Check position data for distance computation
print('\n=== POSITION DATA CHECK ===')
sample_flight = df[df['flight'] == 1].sort_values('time')
print(f'Flight 1 position_x range: {sample_flight["position_x"].min():.6f} - {sample_flight["position_x"].max():.6f}')
print(f'Flight 1 position_y range: {sample_flight["position_y"].min():.6f} - {sample_flight["position_y"].max():.6f}')
print(f'Flight 1 position_z range: {sample_flight["position_z"].min():.6f} - {sample_flight["position_z"].max():.6f}')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# Indian cities with accurate coordinates
cities_data = [
    ('Mumbai', 19.0760, 72.8777, 'Maharashtra'),
    ('Delhi', 28.7041, 77.1025, 'Delhi'),
    ('Kolkata', 22.5726, 88.3639, 'West Bengal'),
    ('Chennai', 13.0827, 80.2707, 'Tamil Nadu'),
    ('Bangalore', 12.9716, 77.5946, 'Karnataka'),
    ('Hyderabad', 17.3850, 78.4867, 'Telangana'),
    ('Ahmedabad', 23.0225, 72.5714, 'Gujarat'),
    ('Pune', 18.5204, 73.8567, 'Maharashtra'),
    ('Surat', 21.1702, 72.8311, 'Gujarat'),
    ('Jaipur', 26.9124, 75.7873, 'Rajasthan'),
    ('Lucknow', 26.8467, 80.9462, 'Uttar Pradesh'),
    ('Kanpur', 26.4499, 80.3319, 'Uttar Pradesh'),
    ('Nagpur', 21.1458, 79.0882, 'Maharashtra'),
    ('Patna', 25.5941, 85.1376, 'Bihar'),
    ('Indore', 22.7196, 75.8577, 'Madhya Pradesh'),
    ('Vadodara', 22.3072, 73.1812, 'Gujarat'),
    ('Bhopal', 23.2599, 77.4126, 'Madhya Pradesh'),
    ('Coimbatore', 11.0168, 76.9558, 'Tamil Nadu'),
    ('Kochi', 9.9312, 76.2673, 'Kerala'),
    ('Visakhapatnam', 17.6868, 83.2185, 'Andhra Pradesh'),
    ('Guwahati', 26.1445, 91.7362, 'Assam'),
    ('Ranchi', 23.3441, 85.3096, 'Jharkhand'),
    ('Thiruvananthapuram', 8.5241, 76.9366, 'Kerala'),
    ('Varanasi', 25.3176, 82.9739, 'Uttar Pradesh'),
    ('Agra', 27.1767, 78.0081, 'Uttar Pradesh'),
    ('Nashik', 19.9975, 73.7898, 'Maharashtra'),
    ('Faridabad', 28.4089, 77.3178, 'Haryana'),
    ('Meerut', 28.9845, 77.7064, 'Uttar Pradesh'),
    ('Rajkot', 22.3039, 70.8022, 'Gujarat'),
    ('Kalyan', 19.2403, 73.1305, 'Maharashtra'),
    ('Vasai', 19.4612, 72.8054, 'Maharashtra'),
    ('Varanasi', 25.3176, 82.9739, 'Uttar Pradesh'),
    ('Srinagar', 34.0837, 74.7973, 'Jammu and Kashmir'),
    ('Aurangabad', 19.8762, 75.3433, 'Maharashtra'),
    ('Dhanbad', 23.7957, 86.4304, 'Jharkhand'),
    ('Amritsar', 31.6340, 74.8723, 'Punjab'),
    ('Allahabad', 25.4358, 81.8463, 'Uttar Pradesh'),
    ('Howrah', 22.5958, 88.2636, 'West Bengal'),
    ('Gwalior', 26.2183, 78.1828, 'Madhya Pradesh'),
    ('Jabalpur', 23.1815, 79.9864, 'Madhya Pradesh'),
    ('Vijayawada', 16.5062, 80.6480, 'Andhra Pradesh'),
    ('Jodhpur', 26.2389, 73.0243, 'Rajasthan'),
    ('Madurai', 9.9252, 78.1198, 'Tamil Nadu'),
    ('Raipur', 21.2514, 81.6296, 'Chhattisgarh'),
    ('Kota', 25.2138, 75.8648, 'Rajasthan'),
    ('Chandigarh', 30.7333, 76.7794, 'Chandigarh'),
    ('Guwahati', 26.1445, 91.7362, 'Assam'),
    ('Solapur', 17.6599, 75.9064, 'Maharashtra'),
    ('Hubli', 15.3647, 75.1240, 'Karnataka'),
    ('Bareilly', 28.3670, 75.4026, 'Uttar Pradesh'),
]

def generate_flood_data(num_records=5000):
    """Generate comprehensive flood dataset with multiple features"""
    
    data = []
    start_date = datetime(2023, 1, 1)
    
    for i in range(num_records):
        # Select random city
        city_name, base_lat, base_lon, state = cities_data[np.random.randint(0, len(cities_data))]
        
        # Add small random variation to coordinates for different locations within city
        latitude = base_lat + np.random.uniform(-0.1, 0.1)
        longitude = base_lon + np.random.uniform(-0.1, 0.1)
        
        # Generate date
        days_offset = np.random.randint(0, 1000)
        date = start_date + timedelta(days=days_offset)
        
        # Generate correlated environmental features
        # Rainfall (0-500mm) - higher rainfall increases flood risk
        rainfall_mm = np.random.gamma(3, 30)
        rainfall_mm = min(rainfall_mm, 500)
        
        # River level (0-20m) - correlated with rainfall
        river_level_m = np.random.gamma(2, 2) + (rainfall_mm / 100) + np.random.normal(0, 1)
        river_level_m = max(0, min(river_level_m, 20))
        
        # Soil moisture (0-100%) - correlated with rainfall
        soil_moisture_percent = min(100, 20 + (rainfall_mm / 5) + np.random.normal(0, 10))
        soil_moisture_percent = max(0, soil_moisture_percent)
        
        # Terrain elevation (0-1000m) - lower elevation = higher risk
        terrain_elevation_m = np.random.gamma(2, 50)
        terrain_elevation_m = min(terrain_elevation_m, 1000)
        
        # Distance to river (0-10km) - closer = higher risk
        distance_to_river_km = np.random.exponential(2)
        distance_to_river_km = min(distance_to_river_km, 10)
        
        # Drainage capacity (0-100%) - lower = higher risk
        drainage_capacity_percent = np.random.beta(3, 2) * 100
        
        # Historical flood count (0-50) - more history = higher risk
        historical_flood_count = int(np.random.poisson(3))
        historical_flood_count = min(historical_flood_count, 50)
        
        # Population density (per sq km)
        population_density = int(np.random.lognormal(7, 2))
        population_density = min(population_density, 50000)
        
        # Urban area percentage (0-100%)
        urban_area_percent = np.random.beta(5, 2) * 100
        
        # Previous 24h rainfall
        prev_24h_rainfall_mm = rainfall_mm * np.random.uniform(0.3, 0.8)
        
        # Previous 48h rainfall
        prev_48h_rainfall_mm = prev_24h_rainfall_mm * np.random.uniform(1.1, 1.5)
        
        # Calculate risk score based on multiple factors
        risk_score = 0
        
        # Rainfall factor (0-40 points)
        if rainfall_mm > 200:
            risk_score += 40
        elif rainfall_mm > 150:
            risk_score += 30
        elif rainfall_mm > 100:
            risk_score += 20
        elif rainfall_mm > 50:
            risk_score += 10
        
        # River level factor (0-25 points)
        if river_level_m > 15:
            risk_score += 25
        elif river_level_m > 10:
            risk_score += 18
        elif river_level_m > 7:
            risk_score += 12
        elif river_level_m > 4:
            risk_score += 6
        
        # Soil moisture factor (0-15 points)
        if soil_moisture_percent > 80:
            risk_score += 15
        elif soil_moisture_percent > 60:
            risk_score += 10
        elif soil_moisture_percent > 40:
            risk_score += 5
        
        # Terrain factor (0-10 points) - lower elevation = more points
        if terrain_elevation_m < 50:
            risk_score += 10
        elif terrain_elevation_m < 100:
            risk_score += 7
        elif terrain_elevation_m < 200:
            risk_score += 4
        
        # Distance to river factor (0-10 points)
        if distance_to_river_km < 1:
            risk_score += 10
        elif distance_to_river_km < 2:
            risk_score += 7
        elif distance_to_river_km < 4:
            risk_score += 4
        
        # Add some randomness
        risk_score += np.random.uniform(-5, 5)
        risk_score = max(0, min(100, risk_score))
        
        # Determine severity level based on risk score
        if risk_score >= 75:
            severity_level = 'Critical'
            water_depth_m = np.random.uniform(3.5, 8.0)
        elif risk_score >= 55:
            severity_level = 'High'
            water_depth_m = np.random.uniform(2.0, 4.0)
        elif risk_score >= 35:
            severity_level = 'Medium'
            water_depth_m = np.random.uniform(0.8, 2.5)
        else:
            severity_level = 'Low'
            water_depth_m = np.random.uniform(0.1, 1.2)
        
        # Population affected (based on severity and density)
        base_affected = int(population_density * np.random.uniform(0.1, 0.5))
        if severity_level == 'Critical':
            population_affected = int(base_affected * np.random.uniform(3, 5))
        elif severity_level == 'High':
            population_affected = int(base_affected * np.random.uniform(2, 3))
        elif severity_level == 'Medium':
            population_affected = int(base_affected * np.random.uniform(1, 2))
        else:
            population_affected = int(base_affected * np.random.uniform(0.3, 1))
        
        # Economic loss (in millions)
        economic_loss_million = population_affected * water_depth_m * np.random.uniform(0.001, 0.005)
        
        data.append({
            'location': city_name,
            'state': state,
            'latitude': round(latitude, 6),
            'longitude': round(longitude, 6),
            'date': date.strftime('%Y-%m-%d'),
            'rainfall_mm': round(rainfall_mm, 2),
            'prev_24h_rainfall_mm': round(prev_24h_rainfall_mm, 2),
            'prev_48h_rainfall_mm': round(prev_48h_rainfall_mm, 2),
            'river_level_m': round(river_level_m, 2),
            'soil_moisture_percent': round(soil_moisture_percent, 2),
            'terrain_elevation_m': round(terrain_elevation_m, 2),
            'distance_to_river_km': round(distance_to_river_km, 2),
            'drainage_capacity_percent': round(drainage_capacity_percent, 2),
            'historical_flood_count': historical_flood_count,
            'population_density': population_density,
            'urban_area_percent': round(urban_area_percent, 2),
            'water_depth_m': round(water_depth_m, 2),
            'population_affected': population_affected,
            'economic_loss_million': round(economic_loss_million, 2),
            'severity_level': severity_level,
            'risk_score': round(risk_score, 2)
        })
    
    return pd.DataFrame(data)

# Generate dataset
print("Generating large flood dataset...")
df = generate_flood_data(num_records=5000)

# Save to CSV
df.to_csv('flood_training_data.csv', index=False)
print(f"Dataset generated with {len(df)} records")
print(f"\nSeverity distribution:")
print(df['severity_level'].value_counts())
print(f"\nDataset saved to flood_training_data.csv")

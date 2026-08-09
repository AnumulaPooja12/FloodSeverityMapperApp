import folium
from folium import plugins
import pandas as pd

class FloodMapGenerator:
    """Generates interactive flood severity maps with threshold-based alerts"""
    
    def __init__(self):
        self.severity_colors = {
            'Low': '#90EE90',
            'Medium': '#FFD700',
            'High': '#FF8C00',
            'Critical': '#DC143C'
        }
        
        self.alert_icons = {
            'normal': 'info-sign',
            'low': 'warning-sign',
            'medium': 'exclamation-sign',
            'high': 'alert',
            'critical': 'remove-sign'
        }
    
    def create_severity_map(self, df, use_predicted=False, show_alerts=False, alerts_data=None):
        """Create an interactive map with flood severity markers"""
        
        if df is None or df.empty:
            return None
        
        # Determine which severity column to use
        if use_predicted and 'predicted_severity' in df.columns:
            severity_col = 'predicted_severity'
        else:
            severity_col = 'severity_level'
        
        # Calculate center coordinates
        center_lat = df['latitude'].mean()
        center_lon = df['longitude'].mean()
        
        # Create base map with zoom level for nationwide view
        flood_map = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=5,
            tiles='OpenStreetMap'
        )
        
        # Create marker cluster for better performance with large datasets
        marker_cluster = plugins.MarkerCluster(name='Flood Locations')
        
        # Track alerts by location if provided
        alert_lookup = {}
        if show_alerts and alerts_data:
            for alert in alerts_data:
                location_key = f"{alert['latitude']:.4f}_{alert['longitude']:.4f}"
                alert_lookup[location_key] = alert
        
        # Add markers for each location
        for idx, row in df.iterrows():
            color = self.severity_colors.get(row[severity_col], 'gray')
            
            # Check if this location has alerts
            location_key = f"{row['latitude']:.4f}_{row['longitude']:.4f}"
            has_alert = location_key in alert_lookup
            alert_info = alert_lookup.get(location_key, None)
            
            # Build popup content
            popup_html = f"""
            <div style="font-family: Arial; min-width: 250px; max-width: 300px;">
                <h4 style="margin: 0 0 10px 0; color: {color};">{row['location']}</h4>
            """
            
            if 'state' in row and pd.notna(row['state']):
                popup_html += f"<p style='margin: 5px 0;'><b>State:</b> {row['state']}</p>"
            
            popup_html += "<table style='width: 100%; font-size: 12px;'>"
            
            # Severity information
            popup_html += f"""
                <tr>
                    <td><b>Severity:</b></td>
                    <td style="color: {color}; font-weight: bold;">{row[severity_col]}</td>
                </tr>
            """
            
            # Add prediction confidence if using predicted values
            if use_predicted and 'prediction_confidence' in row:
                popup_html += f"""
                    <tr>
                        <td><b>Confidence:</b></td>
                        <td>{row['prediction_confidence']:.1%}</td>
                    </tr>
                """
            
            # Add standard metrics
            if 'water_depth_m' in row and pd.notna(row['water_depth_m']):
                popup_html += f"""
                    <tr>
                        <td><b>Water Depth:</b></td>
                        <td>{row['water_depth_m']:.2f}m</td>
                    </tr>
                """
            
            if 'population_affected' in row and pd.notna(row['population_affected']):
                popup_html += f"""
                    <tr>
                        <td><b>Population:</b></td>
                        <td>{row['population_affected']:,.0f}</td>
                    </tr>
                """
            
            if 'rainfall_mm' in row and pd.notna(row['rainfall_mm']):
                popup_html += f"""
                    <tr>
                        <td><b>Rainfall:</b></td>
                        <td>{row['rainfall_mm']:.1f}mm</td>
                    </tr>
                """
            
            if 'river_level_m' in row and pd.notna(row['river_level_m']):
                popup_html += f"""
                    <tr>
                        <td><b>River Level:</b></td>
                        <td>{row['river_level_m']:.1f}m</td>
                    </tr>
                """
            
            popup_html += "</table>"
            
            # Add alert information if available
            if has_alert and alert_info:
                popup_html += f"""
                <div style='margin-top: 10px; padding: 8px; background-color: #fff3cd; border-left: 3px solid {color}; border-radius: 3px;'>
                    <p style='margin: 0; font-weight: bold; color: #856404;'>⚠️ ALERTS ({alert_info['alert_count']})</p>
                """
                
                for alert_detail in alert_info['alerts'][:3]:  # Show top 3 alerts
                    metric_name = alert_detail['metric'].replace('_', ' ').title()
                    popup_html += f"""
                    <p style='margin: 5px 0 0 0; font-size: 11px;'>
                        • {metric_name}: {alert_detail['value']:.1f}<br/>
                        <span style='color: #856404; font-style: italic;'>
                          {alert_detail['recommendation']}
                        </span>
                    </p>
                    """
                
                if alert_info['alert_count'] > 3:
                    popup_html += f"<p style='margin: 5px 0 0 0; font-size: 11px;'>... and {alert_info['alert_count'] - 3} more</p>"
                
                popup_html += "</div>"
            
            popup_html += "</div>"
            
            # Calculate marker size based on water depth or severity
            base_radius = 8
            if 'water_depth_m' in row and pd.notna(row['water_depth_m']):
                radius = base_radius + (row['water_depth_m'] * 1.5)
            else:
                severity_multipliers = {'Low': 1, 'Medium': 1.5, 'High': 2, 'Critical': 2.5}
                multiplier = severity_multipliers.get(row[severity_col], 1)
                radius = base_radius * multiplier
            
            # Add circle marker with alert indicator
            circle_marker = folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=radius,
                popup=folium.Popup(popup_html, max_width=300),
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7 if not has_alert else 0.9,
                weight=2 if not has_alert else 3
            )
            
            marker_cluster.add_child(circle_marker)
            
            # Add special marker for locations with critical/high alerts
            if has_alert and alert_info['max_severity'] in ['critical', 'high']:
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    icon=folium.Icon(color='red', icon='exclamation-triangle', prefix='fa'),
                    popup=folium.Popup(popup_html, max_width=300)
                ).add_to(flood_map)
        
        marker_cluster.add_to(flood_map)
        
        # Add legend
        legend_html = f'''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; 
                    background-color: white; 
                    border: 2px solid grey; 
                    border-radius: 5px;
                    padding: 10px;
                    font-family: Arial;
                    z-index: 9999;">
            <h4 style="margin: 0 0 10px 0;">Severity Levels</h4>
            <div style="margin: 5px 0;">
                <span style="background-color: {self.severity_colors['Critical']}; 
                             width: 20px; height: 20px; 
                             display: inline-block; 
                             margin-right: 5px;
                             border-radius: 50%;"></span>
                Critical
            </div>
            <div style="margin: 5px 0;">
                <span style="background-color: {self.severity_colors['High']}; 
                             width: 20px; height: 20px; 
                             display: inline-block; 
                             margin-right: 5px;
                             border-radius: 50%;"></span>
                High
            </div>
            <div style="margin: 5px 0;">
                <span style="background-color: {self.severity_colors['Medium']}; 
                             width: 20px; height: 20px; 
                             display: inline-block; 
                             margin-right: 5px;
                             border-radius: 50%;"></span>
                Medium
            </div>
            <div style="margin: 5px 0;">
                <span style="background-color: {self.severity_colors['Low']}; 
                             width: 20px; height: 20px; 
                             display: inline-block; 
                             margin-right: 5px;
                             border-radius: 50%;"></span>
                Low
            </div>
        '''
        
        if show_alerts:
            legend_html += '''
            <hr style="margin: 10px 0;">
            <div style="margin: 5px 0;">
                <i class="fa fa-exclamation-triangle" style="color: red; margin-right: 5px;"></i>
                Alert Active
            </div>
            '''
        
        legend_html += '</div>'
        
        flood_map.get_root().html.add_child(folium.Element(legend_html))
        
        # Add fullscreen option
        plugins.Fullscreen().add_to(flood_map)
        
        # Add layer control
        folium.LayerControl().add_to(flood_map)
        
        return flood_map
    
    def create_heatmap(self, df, intensity_metric='water_depth_m'):
        """Create a heatmap showing flood intensity"""
        
        if df is None or df.empty:
            return None
        
        # Calculate center coordinates
        center_lat = df['latitude'].mean()
        center_lon = df['longitude'].mean()
        
        # Create base map
        heat_map = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=5,
            tiles='OpenStreetMap'
        )
        
        # Prepare heat data
        heat_data = []
        
        if intensity_metric in df.columns:
            for idx, row in df.iterrows():
                heat_data.append([
                    row['latitude'],
                    row['longitude'],
                    float(row[intensity_metric]) if pd.notna(row[intensity_metric]) else 0
                ])
        else:
            # Fallback to simple heatmap
            for idx, row in df.iterrows():
                heat_data.append([row['latitude'], row['longitude'], 1])
        
        # Add heatmap layer
        plugins.HeatMap(
            heat_data,
            radius=15,
            blur=25,
            max_zoom=13,
            gradient={
                0.0: 'blue',
                0.3: 'lime',
                0.5: 'yellow',
                0.7: 'orange',
                1.0: 'red'
            }
        ).add_to(heat_map)
        
        # Add fullscreen option
        plugins.Fullscreen().add_to(heat_map)
        
        return heat_map
    
    def create_alert_map(self, alerts_data):
        """Create a map specifically for visualizing alerts"""
        
        if not alerts_data:
            return None
        
        # Calculate center from alerts
        lats = [alert['latitude'] for alert in alerts_data]
        lons = [alert['longitude'] for alert in alerts_data]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
        
        # Create base map
        alert_map = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=5,
            tiles='OpenStreetMap'
        )
        
        # Add markers for each alert
        for alert in alerts_data:
            color = self.severity_colors.get(alert['max_severity'], 'gray')
            
            # Build popup
            popup_html = f"""
            <div style="font-family: Arial; min-width: 250px;">
                <h4 style="margin: 0 0 10px 0; color: {color};">
                    ⚠️ {alert['max_severity'].upper()} ALERT
                </h4>
                <p><b>Location:</b> {alert['location']}</p>
                <p><b>State:</b> {alert.get('state', 'Unknown')}</p>
                <p><b>Alerts:</b> {alert['alert_count']}</p>
                <p><b>Time:</b> {alert['timestamp']}</p>
                <hr style="margin: 10px 0;">
                <p style="font-weight: bold;">Threshold Violations:</p>
            """
            
            for alert_detail in alert['alerts']:
                popup_html += f"""
                <div style="margin: 5px 0; padding: 5px; background-color: #f8f9fa; border-radius: 3px;">
                    <p style="margin: 0; font-weight: bold; color: {color};">
                        {alert_detail['metric'].replace('_', ' ').title()}
                    </p>
                    <p style="margin: 3px 0; font-size: 12px;">
                        Value: {alert_detail['value']:.2f} (Threshold: {alert_detail['threshold']})
                    </p>
                    <p style="margin: 3px 0; font-size: 11px; font-style: italic; color: #666;">
                        {alert_detail['recommendation']}
                    </p>
                </div>
                """
            
            popup_html += "</div>"
            
            # Add marker
            folium.Marker(
                location=[alert['latitude'], alert['longitude']],
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(
                    color='red' if alert['max_severity'] in ['critical', 'high'] else 'orange',
                    icon='exclamation-triangle',
                    prefix='fa'
                )
            ).add_to(alert_map)
            
            # Add circle to show severity
            folium.CircleMarker(
                location=[alert['latitude'], alert['longitude']],
                radius=10 + (alert['alert_count'] * 2),
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.3,
                weight=2
            ).add_to(alert_map)
        
        # Add fullscreen option
        plugins.Fullscreen().add_to(alert_map)
        
        return alert_map

import pandas as pd
import numpy as np
from datetime import datetime

class ThresholdBasedAlertSystem:
    """Threshold-based alert system for flood monitoring"""
    
    def __init__(self):
        # Default threshold configurations
        self.thresholds = {
            'rainfall_mm': {
                'low': 50,
                'medium': 100,
                'high': 150,
                'critical': 200
            },
            'water_depth_m': {
                'low': 1.0,
                'medium': 2.0,
                'high': 3.5,
                'critical': 5.0
            },
            'river_level_m': {
                'low': 5,
                'medium': 8,
                'high': 12,
                'critical': 15
            },
            'soil_moisture_percent': {
                'low': 40,
                'medium': 60,
                'high': 80,
                'critical': 90
            },
            'population_affected': {
                'low': 5000,
                'medium': 10000,
                'high': 20000,
                'critical': 50000
            },
            'risk_score': {
                'low': 35,
                'medium': 55,
                'high': 75,
                'critical': 90
            }
        }
        
        self.alert_messages = {
            'low': 'Monitor the situation',
            'medium': 'Prepare for possible evacuation',
            'high': 'Evacuate vulnerable areas',
            'critical': 'Immediate evacuation required'
        }
        
        self.alert_colors = {
            'low': '#90EE90',
            'medium': '#FFD700',
            'high': '#FF8C00',
            'critical': '#DC143C'
        }
    
    def set_threshold(self, metric, level, value):
        """Set custom threshold for a metric"""
        
        if metric not in self.thresholds:
            raise ValueError(f"Unknown metric: {metric}")
        
        if level not in ['low', 'medium', 'high', 'critical']:
            raise ValueError(f"Invalid level: {level}")
        
        self.thresholds[metric][level] = value
    
    def get_thresholds(self):
        """Get current threshold configuration"""
        return self.thresholds.copy()
    
    def check_threshold(self, metric, value):
        """Check which threshold level is exceeded for a given metric value"""
        
        if metric not in self.thresholds:
            return None
        
        thresholds = self.thresholds[metric]
        
        if value >= thresholds['critical']:
            return 'critical'
        elif value >= thresholds['high']:
            return 'high'
        elif value >= thresholds['medium']:
            return 'medium'
        elif value >= thresholds['low']:
            return 'low'
        else:
            return 'normal'
    
    def generate_alerts(self, df):
        """Generate alerts for all locations in the dataframe"""
        
        alerts = []
        
        for idx, row in df.iterrows():
            location_alerts = self.check_location_alerts(row)
            
            if location_alerts['alerts']:
                alerts.append(location_alerts)
        
        return alerts
    
    def check_location_alerts(self, location_data):
        """Check all threshold alerts for a single location"""
        
        alert_info = {
            'location': location_data.get('location', 'Unknown'),
            'state': location_data.get('state', 'Unknown'),
            'latitude': location_data.get('latitude', 0),
            'longitude': location_data.get('longitude', 0),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'alerts': [],
            'max_severity': 'normal',
            'alert_count': 0
        }
        
        severity_priority = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1, 'normal': 0}
        max_priority = 0
        
        # Check each metric
        for metric in self.thresholds.keys():
            if metric in location_data and pd.notna(location_data[metric]):
                value = location_data[metric]
                level = self.check_threshold(metric, value)
                
                if level and level != 'normal':
                    threshold_value = self.thresholds[metric][level]
                    
                    alert_info['alerts'].append({
                        'metric': metric,
                        'value': float(value),
                        'threshold': threshold_value,
                        'level': level,
                        'message': f"{metric.replace('_', ' ').title()}: {value:.2f} (Threshold: {threshold_value})",
                        'recommendation': self.alert_messages[level]
                    })
                    
                    # Track maximum severity
                    if severity_priority[level] > max_priority:
                        max_priority = severity_priority[level]
                        alert_info['max_severity'] = level
        
        alert_info['alert_count'] = len(alert_info['alerts'])
        
        return alert_info
    
    def get_critical_alerts(self, alerts):
        """Filter only critical and high severity alerts"""
        
        critical_alerts = [
            alert for alert in alerts
            if alert['max_severity'] in ['critical', 'high']
        ]
        
        return sorted(critical_alerts, key=lambda x: x['alert_count'], reverse=True)
    
    def get_alert_summary(self, alerts):
        """Generate summary statistics for alerts"""
        
        if not alerts:
            return {
                'total_alerts': 0,
                'critical_count': 0,
                'high_count': 0,
                'medium_count': 0,
                'low_count': 0
            }
        
        summary = {
            'total_alerts': len(alerts),
            'critical_count': sum(1 for a in alerts if a['max_severity'] == 'critical'),
            'high_count': sum(1 for a in alerts if a['max_severity'] == 'high'),
            'medium_count': sum(1 for a in alerts if a['max_severity'] == 'medium'),
            'low_count': sum(1 for a in alerts if a['max_severity'] == 'low')
        }
        
        return summary
    
    def format_alert_message(self, alert):
        """Format alert information for display"""
        
        location_name = alert['location']
        state_name = alert.get('state', '')
        severity = alert['max_severity']
        alert_count = alert['alert_count']
        
        message = f"🚨 {severity.upper()} ALERT: {location_name}"
        if state_name:
            message += f", {state_name}"
        
        message += f"\n📊 {alert_count} threshold(s) exceeded\n"
        message += f"⏰ {alert['timestamp']}\n\n"
        
        message += "Details:\n"
        for alert_detail in alert['alerts']:
            message += f"  • {alert_detail['message']}\n"
            message += f"    → {alert_detail['recommendation']}\n"
        
        return message
    
    def export_alerts_to_dataframe(self, alerts):
        """Export alerts to a structured DataFrame"""
        
        if not alerts:
            return pd.DataFrame()
        
        alert_records = []
        
        for alert in alerts:
            base_record = {
                'location': alert['location'],
                'state': alert.get('state', ''),
                'latitude': alert['latitude'],
                'longitude': alert['longitude'],
                'timestamp': alert['timestamp'],
                'severity': alert['max_severity'],
                'alert_count': alert['alert_count']
            }
            
            # Add each metric that triggered an alert
            for alert_detail in alert['alerts']:
                record = base_record.copy()
                record['metric'] = alert_detail['metric']
                record['value'] = alert_detail['value']
                record['threshold'] = alert_detail['threshold']
                record['level'] = alert_detail['level']
                record['recommendation'] = alert_detail['recommendation']
                alert_records.append(record)
        
        return pd.DataFrame(alert_records)
    
    def get_alert_statistics_by_metric(self, alerts):
        """Get statistics about which metrics are triggering most alerts"""
        
        if not alerts:
            return pd.DataFrame()
        
        metric_counts = {}
        
        for alert in alerts:
            for alert_detail in alert['alerts']:
                metric = alert_detail['metric']
                level = alert_detail['level']
                
                if metric not in metric_counts:
                    metric_counts[metric] = {'total': 0, 'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
                
                metric_counts[metric]['total'] += 1
                metric_counts[metric][level] += 1
        
        stats_df = pd.DataFrame.from_dict(metric_counts, orient='index')
        stats_df = stats_df.reset_index().rename(columns={'index': 'metric'})
        stats_df = stats_df.sort_values('total', ascending=False)
        
        return stats_df
    
    def configure_from_percentiles(self, df, percentiles={'low': 25, 'medium': 50, 'high': 75, 'critical': 90}):
        """Automatically configure thresholds based on data percentiles"""
        
        for metric in self.thresholds.keys():
            if metric in df.columns:
                for level, percentile in percentiles.items():
                    threshold_value = df[metric].quantile(percentile / 100)
                    self.thresholds[metric][level] = float(threshold_value)
        
        print("Thresholds configured based on data percentiles")
        return self.thresholds


# Helper function to create and configure alert system
def create_alert_system(custom_thresholds=None):
    """Create and optionally configure an alert system"""
    
    alert_system = ThresholdBasedAlertSystem()
    
    if custom_thresholds:
        for metric, levels in custom_thresholds.items():
            for level, value in levels.items():
                alert_system.set_threshold(metric, level, value)
    
    return alert_system


if __name__ == "__main__":
    # Example usage
    print("Threshold-Based Alert System")
    print("="*60)
    
    alert_system = ThresholdBasedAlertSystem()
    
    print("\nDefault Thresholds:")
    for metric, levels in alert_system.get_thresholds().items():
        print(f"\n{metric}:")
        for level, value in levels.items():
            print(f"  {level}: {value}")
    
    # Test with sample data
    sample_location = {
        'location': 'Mumbai',
        'state': 'Maharashtra',
        'latitude': 19.0760,
        'longitude': 72.8777,
        'rainfall_mm': 220,
        'water_depth_m': 4.5,
        'river_level_m': 16,
        'soil_moisture_percent': 85,
        'population_affected': 35000
    }
    
    print("\n" + "="*60)
    print("Testing with sample location:")
    print("="*60)
    
    alert_info = alert_system.check_location_alerts(sample_location)
    print(alert_system.format_alert_message(alert_info))

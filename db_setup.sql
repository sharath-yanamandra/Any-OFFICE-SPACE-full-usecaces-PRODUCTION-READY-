-- Script 4: setup_datacenter_config.sql
-- File Path: setup_datacenter_config.sql
--
-- Datacenter Monitoring System - Database Setup Script
--
-- This script sets up sample data for datacenter monitoring including:
-- 1. Sample users (datacenter administrators and operators)
-- 2. Sample datacenters (facilities)
-- 3. Sample cameras with different monitoring types
-- 4. Sample zones for different areas
-- 5. Sample rules for various event types

-- Insert sample users for datacenter monitoring
INSERT INTO users (user_id, email, full_name, hashed_password, role, is_active, is_verified)
VALUES 
('dc-admin-001', 'admin@ctrls.com', 'Datacenter Administrator', '$2b$12$encrypted_password_hash', 'admin', TRUE, TRUE),
('dc-operator-001', 'operator1@ctrls.com', 'Security Operator 1', '$2b$12$encrypted_password_hash', 'operator', TRUE, TRUE),
('dc-operator-002', 'operator2@ctrls.com', 'Security Operator 2', '$2b$12$encrypted_password_hash', 'operator', TRUE, TRUE),
('dc-viewer-001', 'viewer@ctrls.com', 'Security Viewer', '$2b$12$encrypted_password_hash', 'viewer', TRUE, TRUE);

-- Insert sample datacenters (facilities)
INSERT INTO datacenters (user_id, name, description, location, address, coordinates, facility_type, capacity_info, contact_info, status)
VALUES 
('dc-admin-001', 'CtrlS Mumbai DC1', 'Primary datacenter facility in Mumbai', 'Mumbai, Maharashtra', 
 'Plot No. 123, MIDC Industrial Area, Andheri East, Mumbai - 400093', '19.1136,72.8697', 'tier3',
 '{"rack_count": 500, "power_capacity_kw": 2000, "cooling_capacity_tons": 300, "floor_area_sqft": 50000}',
 '{"facility_manager": "John Doe", "emergency_contact": "+91-9876543210", "security_head": "Jane Smith"}', 'active'),

('dc-admin-001', 'CtrlS Hyderabad DC2', 'Secondary datacenter facility in Hyderabad', 'Hyderabad, Telangana',
 'Survey No. 456, HITEC City, Madhapur, Hyderabad - 500081', '17.4485,78.3908', 'tier3',
 '{"rack_count": 400, "power_capacity_kw": 1500, "cooling_capacity_tons": 250, "floor_area_sqft": 40000}',
 '{"facility_manager": "Raj Kumar", "emergency_contact": "+91-9876543211", "security_head": "Priya Sharma"}', 'active'),

('dc-admin-001', 'CtrlS Chennai DC3', 'Tertiary datacenter facility in Chennai', 'Chennai, Tamil Nadu',
 'No. 789, IT Expressway, OMR, Chennai - 600119', '12.8956,80.2210', 'tier2',
 '{"rack_count": 300, "power_capacity_kw": 1000, "cooling_capacity_tons": 200, "floor_area_sqft": 30000}',
 '{"facility_manager": "Suresh Babu", "emergency_contact": "+91-9876543212", "security_head": "Kavya Reddy"}', 'active');

-- Get datacenter IDs for reference
SET @dc1 = (SELECT datacenter_id FROM datacenters WHERE name = 'CtrlS Mumbai DC1' AND user_id = 'dc-admin-001');
SET @dc2 = (SELECT datacenter_id FROM datacenters WHERE name = 'CtrlS Hyderabad DC2' AND user_id = 'dc-admin-001');
SET @dc3 = (SELECT datacenter_id FROM datacenters WHERE name = 'CtrlS Chennai DC3' AND user_id = 'dc-admin-001');

-- Insert sample cameras for Mumbai DC1
INSERT INTO cameras (datacenter_id, name, stream_url, camera_type, location_details, status, metadata, installation_date)
VALUES
-- Entry monitoring cameras
(@dc1, 'Main Entry Camera 1', 'rtsp://admin:password@192.168.1.101:554/ch0_0.264', 'dc_entry_monitor', 
 '{"floor": "Ground", "location": "Main Entrance", "direction": "inbound"}', 'active',
 '{"resolution": "1920x1080", "fps": 30, "night_vision": true, "ptz": false, "activity_level": "high"}', '2024-01-15'),

(@dc1, 'Main Entry Camera 2', 'rtsp://admin:password@192.168.1.102:554/ch0_0.264', 'dc_entry_monitor',
 '{"floor": "Ground", "location": "Main Entrance", "direction": "outbound"}', 'active',
 '{"resolution": "1920x1080", "fps": 30, "night_vision": true, "ptz": false, "activity_level": "high"}', '2024-01-15'),

-- Server room cameras
(@dc1, 'Server Room A Camera', 'rtsp://admin:password@192.168.1.103:554/ch0_0.264', 'dc_server_room',
 '{"floor": "1st", "location": "Server Room A", "rack_rows": ["A1-A10"], "zone": "high_security"}', 'active',
 '{"resolution": "1920x1080", "fps": 15, "night_vision": true, "ptz": true, "activity_level": "medium"}', '2024-01-16'),

(@dc1, 'Server Room B Camera', 'rtsp://admin:password@192.168.1.104:554/ch0_0.264', 'dc_server_room',
 '{"floor": "1st", "location": "Server Room B", "rack_rows": ["B1-B8"], "zone": "high_security"}', 'active',
 '{"resolution": "1920x1080", "fps": 15, "night_vision": true, "ptz": true, "activity_level": "medium"}', '2024-01-16'),

-- Corridor cameras
(@dc1, 'Corridor Camera 1', 'rtsp://admin:password@192.168.1.105:554/ch0_0.264', 'dc_corridor',
 '{"floor": "1st", "location": "Main Corridor", "section": "North Wing"}', 'active',
 '{"resolution": "1920x1080", "fps": 10, "night_vision": true, "ptz": false, "activity_level": "low"}', '2024-01-17'),

(@dc1, 'Corridor Camera 2', 'rtsp://admin:password@192.168.1.106:554/ch0_0.264', 'dc_corridor',
 '{"floor": "2nd", "location": "Main Corridor", "section": "South Wing"}', 'active',
 '{"resolution": "1920x1080", "fps": 10, "night_vision": true, "ptz": false, "activity_level": "low"}', '2024-01-17'),

-- Perimeter cameras
(@dc1, 'Perimeter Camera 1', 'rtsp://admin:password@192.168.1.107:554/ch0_0.264', 'dc_perimeter',
 '{"floor": "Ground", "location": "North Perimeter", "coverage_angle": 120}', 'active',
 '{"resolution": "1920x1080", "fps": 15, "night_vision": true, "ptz": true, "activity_level": "medium"}', '2024-01-18'),

-- Critical zone camera
(@dc1, 'UPS Room Camera', 'rtsp://admin:password@192.168.1.108:554/ch0_0.264', 'dc_critical_zone',
 '{"floor": "Ground", "location": "UPS Room", "zone": "critical_infrastructure"}', 'active',
 '{"resolution": "1920x1080", "fps": 20, "night_vision": true, "ptz": false, "activity_level": "high"}', '2024-01-19');

-- Insert sample cameras for Hyderabad DC2
INSERT INTO cameras (datacenter_id, name, stream_url, camera_type, location_details, status, metadata, installation_date)
VALUES
(@dc2, 'HYD Entry Camera', 'rtsp://admin:password@192.168.2.101:554/ch0_0.264', 'dc_entry_monitor',
 '{"floor": "Ground", "location": "Main Entrance", "direction": "bidirectional"}', 'active',
 '{"resolution": "1920x1080", "fps": 30, "night_vision": true, "ptz": false, "activity_level": "high"}', '2024-02-01'),

(@dc2, 'HYD Server Room Camera', 'rtsp://admin:password@192.168.2.102:554/ch0_0.264', 'dc_server_room',
 '{"floor": "1st", "location": "Main Server Hall", "rack_rows": ["C1-C15"], "zone": "high_security"}', 'active',
 '{"resolution": "1920x1080", "fps": 15, "night_vision": true, "ptz": true, "activity_level": "medium"}', '2024-02-01'),

(@dc2, 'HYD Common Area Camera', 'rtsp://admin:password@192.168.2.103:554/ch0_0.264', 'dc_common_area',
 '{"floor": "Ground", "location": "Reception Area", "coverage": "waiting_area"}', 'active',
 '{"resolution": "1920x1080", "fps": 10, "night_vision": false, "ptz": false, "activity_level": "low"}', '2024-02-02');

-- Insert sample cameras for Chennai DC3
INSERT INTO cameras (datacenter_id, name, stream_url, camera_type, location_details, status, metadata, installation_date)
VALUES
(@dc3, 'CHN Entry Camera', 'rtsp://admin:password@192.168.3.101:554/ch0_0.264', 'dc_entry_monitor',
 '{"floor": "Ground", "location": "Security Checkpoint", "direction": "inbound"}', 'active',
 '{"resolution": "1920x1080", "fps": 30, "night_vision": true, "ptz": false, "activity_level": "high"}', '2024-03-01'),

(@dc3, 'CHN Server Camera', 'rtsp://admin:password@192.168.3.102:554/ch0_0.264', 'dc_server_room',
 '{"floor": "1st", "location": "Compute Hall", "rack_rows": ["D1-D12"], "zone": "restricted"}', 'active',
 '{"resolution": "1920x1080", "fps": 15, "night_vision": true, "ptz": false, "activity_level": "medium"}', '2024-03-01');

-- Get camera IDs for zone and rule setup
SET @cam1 = (SELECT camera_id FROM cameras WHERE name = 'Main Entry Camera 1' AND datacenter_id = @dc1);
SET @cam2 = (SELECT camera_id FROM cameras WHERE name = 'Main Entry Camera 2' AND datacenter_id = @dc1);
SET @cam3 = (SELECT camera_id FROM cameras WHERE name = 'Server Room A Camera' AND datacenter_id = @dc1);
SET @cam4 = (SELECT camera_id FROM cameras WHERE name = 'Server Room B Camera' AND datacenter_id = @dc1);
SET @cam5 = (SELECT camera_id FROM cameras WHERE name = 'Corridor Camera 1' AND datacenter_id = @dc1);
SET @cam6 = (SELECT camera_id FROM cameras WHERE name = 'Corridor Camera 2' AND datacenter_id = @dc1);
SET @cam7 = (SELECT camera_id FROM cameras WHERE name = 'Perimeter Camera 1' AND datacenter_id = @dc1);
SET @cam8 = (SELECT camera_id FROM cameras WHERE name = 'UPS Room Camera' AND datacenter_id = @dc1);

-- Insert sample zones for Mumbai DC1 cameras
INSERT INTO zones (camera_id, name, zone_type, polygon_coordinates, security_level, access_requirements, monitoring_rules)
VALUES
-- Entry zones
(@cam1, 'Main Entry Inbound Zone', 'entry_zone', '[[100, 100], [500, 100], [500, 400], [100, 400]]', 'restricted',
 '{"badge_required": true, "escort_required": false, "time_restrictions": "24x7"}',
 '{"max_occupancy": 3, "loitering_threshold": 30, "tailgating_detection": true}'),

(@cam2, 'Main Entry Outbound Zone', 'entry_zone', '[[600, 100], [1000, 100], [1000, 400], [600, 400]]', 'restricted',
 '{"badge_required": true, "escort_required": false, "time_restrictions": "24x7"}',
 '{"max_occupancy": 3, "loitering_threshold": 30, "tailgating_detection": true}'),

-- Server room zones
(@cam3, 'Server Room A Zone', 'server_zone', '[[0, 0], [1920, 0], [1920, 1080], [0, 1080]]', 'high_security',
 '{"badge_required": true, "escort_required": true, "ppe_required": ["hard_hat", "safety_vest"], "time_restrictions": "business_hours"}',
 '{"max_occupancy": 5, "ppe_detection": true, "unauthorized_access_alert": true}'),

(@cam4, 'Server Room B Zone', 'server_zone', '[[0, 0], [1920, 0], [1920, 1080], [0, 1080]]', 'high_security',
 '{"badge_required": true, "escort_required": true, "ppe_required": ["hard_hat", "safety_vest"], "time_restrictions": "business_hours"}',
 '{"max_occupancy": 5, "ppe_detection": true, "unauthorized_access_alert": true}'),

-- Common zones
(@cam5, 'North Corridor Zone', 'common_zone', '[[200, 200], [1720, 200], [1720, 880], [200, 880]]', 'restricted',
 '{"badge_required": true, "escort_required": false}',
 '{"max_occupancy": 10, "loitering_threshold": 300, "people_counting": true}'),

(@cam6, 'South Corridor Zone', 'common_zone', '[[200, 200], [1720, 200], [1720, 880], [200, 880]]', 'restricted',
 '{"badge_required": true, "escort_required": false}',
 '{"max_occupancy": 10, "loitering_threshold": 300, "people_counting": true}'),

-- Perimeter zone
(@cam7, 'North Perimeter Zone', 'perimeter_zone', '[[0, 0], [1920, 0], [1920, 1080], [0, 1080]]', 'restricted',
 '{"after_hours_alert": true}',
 '{"intrusion_detection": true, "motion_sensitivity": "high"}'),

-- Critical zone
(@cam8, 'UPS Room Critical Zone', 'critical_zone', '[[150, 150], [1770, 150], [1770, 930], [150, 930]]', 'critical',
 '{"badge_required": true, "escort_required": true, "ppe_required": ["hard_hat", "safety_vest", "safety_glasses"], "special_authorization": true}',
 '{"max_occupancy": 2, "ppe_detection": true, "unauthorized_access_alert": true, "equipment_tampering_detection": true}');

-- Insert monitoring rules for different event types
INSERT INTO rules (camera_id, name, description, event_type, severity, parameters, schedule, notification_settings, enabled)
VALUES
-- Entry monitoring rules
(@cam1, 'Tailgating Detection', 'Detect multiple people entering with single badge scan', 'tailgating', 'high',
 '{"max_people_per_entry": 1, "time_window_seconds": 10, "confidence_threshold": 0.8}',
 '{"active_24x7": true}',
 '{"sms_alert": true, "email_alert": true, "dashboard_alert": true}', TRUE),

(@cam1, 'Entry Zone Intrusion', 'Detect unauthorized access to entry area', 'intrusion', 'high',
 '{"detection_confidence": 0.7, "ignore_authorized_hours": false}',
 '{"active_24x7": true}',
 '{"sms_alert": true, "email_alert": true, "dashboard_alert": true}', TRUE),

-- Server room rules
(@cam3, 'PPE Compliance Check', 'Ensure PPE compliance in server room', 'ppe_violation', 'medium',
 '{"required_ppe": ["hard_hat", "safety_vest"], "detection_confidence": 0.7, "grace_period_seconds": 30}',
 '{"business_hours_only": true, "weekdays": [1,2,3,4,5]}',
 '{"email_alert": true, "dashboard_alert": true}', TRUE),

(@cam3, 'Server Room Intrusion', 'Detect unauthorized access to server room', 'intrusion', 'critical',
 '{"detection_confidence": 0.8, "immediate_alert": true}',
 '{"active_24x7": true}',
 '{"sms_alert": true, "email_alert": true, "dashboard_alert": true, "security_alert": true}', TRUE),

(@cam4, 'Server Room Occupancy', 'Monitor server room occupancy levels', 'people_counting', 'low',
 '{"max_occupancy": 5, "warning_threshold": 3, "alert_threshold": 5}',
 '{"business_hours_only": true}',
 '{"dashboard_alert": true}', TRUE),

-- Corridor rules
(@cam5, 'Corridor Loitering', 'Detect loitering in corridor areas', 'loitering', 'medium',
 '{"max_stationary_time_seconds": 300, "movement_threshold_meters": 1.0, "detection_confidence": 0.6}',
 '{"active_24x7": true}',
 '{"email_alert": true, "dashboard_alert": true}', TRUE),

(@cam6, 'Corridor People Counting', 'Monitor corridor occupancy', 'people_counting', 'low',
 '{"max_occupancy": 10, "warning_threshold": 7, "alert_threshold": 10}',
 '{"active_24x7": true}',
 '{"dashboard_alert": true}', TRUE),

-- Perimeter rules
(@cam7, 'Perimeter Intrusion', 'Detect perimeter breaches', 'intrusion', 'critical',
 '{"detection_confidence": 0.8, "after_hours_sensitivity": "high", "motion_threshold": 0.3}',
 '{"active_24x7": true, "enhanced_after_hours": true}',
 '{"sms_alert": true, "email_alert": true, "dashboard_alert": true, "security_alert": true}', TRUE),

-- Critical zone rules
(@cam8, 'UPS Room Access Control', 'Monitor access to critical UPS room', 'unauthorized_access', 'critical',
 '{"detection_confidence": 0.9, "require_escort": true, "max_occupancy": 2}',
 '{"active_24x7": true}',
 '{"sms_alert": true, "email_alert": true, "dashboard_alert": true, "security_alert": true, "management_alert": true}', TRUE),

(@cam8, 'UPS Room PPE Compliance', 'Ensure full PPE compliance in UPS room', 'ppe_violation', 'high',
 '{"required_ppe": ["hard_hat", "safety_vest", "safety_glasses"], "detection_confidence": 0.8, "grace_period_seconds": 15}',
 '{"active_24x7": true}',
 '{"sms_alert": true, "email_alert": true, "dashboard_alert": true}', TRUE);

-- Add some camera tamper detection rules for all cameras
INSERT INTO rules (camera_id, name, description, event_type, severity, parameters, schedule, notification_settings, enabled)
SELECT 
    camera_id,
    CONCAT(name, ' - Tamper Detection'),
    'Detect camera obstruction or tampering',
    'camera_tamper',
    'high',
    '{"frame_diff_threshold": 0.8, "obstruction_threshold": 0.9, "check_interval_seconds": 60}',
    '{"active_24x7": true}',
    '{"sms_alert": true, "email_alert": true, "dashboard_alert": true, "technical_alert": true}',
    TRUE
FROM cameras 
WHERE datacenter_id = @dc1;

-- Insert some sample access logs for demonstration
INSERT INTO access_logs (datacenter_id, camera_id, zone_id, person_id, access_type, timestamp, detection_confidence, additional_data)
VALUES
(@dc1, @cam1, (SELECT zone_id FROM zones WHERE name = 'Main Entry Inbound Zone'), 'EMP001', 'entry', DATE_SUB(NOW(), INTERVAL 2 HOUR), 0.95,
 '{"badge_scan": true, "ppe_status": {"hard_hat": false, "safety_vest": false}, "duration_seconds": 3}'),

(@dc1, @cam3, (SELECT zone_id FROM zones WHERE name = 'Server Room A Zone'), 'EMP001', 'entry', DATE_SUB(NOW(), INTERVAL 1 HOUR), 0.92,
 '{"badge_scan": true, "ppe_status": {"hard_hat": true, "safety_vest": true}, "escort_present": true, "duration_seconds": 1800}'),

(@dc1, @cam3, (SELECT zone_id FROM zones WHERE name = 'Server Room A Zone'), 'EMP001', 'exit', DATE_SUB(NOW(), INTERVAL 30 MINUTE), 0.89,
 '{"badge_scan": true, "ppe_status": {"hard_hat": true, "safety_vest": true}, "duration_seconds": 5}');

-- Create some test events for demonstration
INSERT INTO events (event_id, rule_id, camera_id, zone_id, timestamp, event_type, severity, detection_data, status)
VALUES
('evt-001', (SELECT rule_id FROM rules WHERE name = 'Tailgating Detection' LIMIT 1), @cam1, 
 (SELECT zone_id FROM zones WHERE name = 'Main Entry Inbound Zone'), DATE_SUB(NOW(), INTERVAL 3 HOUR),
 'tailgating', 'high',
 '{"people_count": 2, "confidence": 0.87, "detection_details": {"person1": {"bbox": [100, 150, 200, 400]}, "person2": {"bbox": [220, 160, 320, 410]}}}',
 'resolved'),

('evt-002', (SELECT rule_id FROM rules WHERE name = 'PPE Compliance Check' LIMIT 1), @cam3,
 (SELECT zone_id FROM zones WHERE name = 'Server Room A Zone'), DATE_SUB(NOW(), INTERVAL 1 HOUR),
 'ppe_violation', 'medium',
 '{"person_count": 1, "confidence": 0.82, "missing_ppe": ["hard_hat"], "detection_details": {"person1": {"bbox": [300, 200, 400, 500], "ppe_status": {"hard_hat": false, "safety_vest": true}}}}',
 'acknowledged'),

('evt-003', (SELECT rule_id FROM rules WHERE name = 'Perimeter Intrusion' LIMIT 1), @cam7,
 (SELECT zone_id FROM zones WHERE name = 'North Perimeter Zone'), DATE_SUB(NOW(), INTERVAL 30 MINUTE),
 'intrusion', 'critical',
 '{"people_count": 1, "confidence": 0.91, "after_hours": true, "detection_details": {"person1": {"bbox": [500, 300, 600, 700]}}}',
 'new');
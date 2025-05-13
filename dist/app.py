from flask import Flask, render_template
from flask_socketio import SocketIO
import asyncio
import nest_asyncio
from bleak import BleakScanner, BleakClient
import threading
import json
import numpy as np
from scipy.signal import welch
from scipy.interpolate import interp1d
from datetime import datetime
import time

app = Flask(__name__)
socketio = SocketIO(app)
nest_asyncio.apply()

HR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"

# Store heart rate data for HRV calculation
heart_rate_data = []
last_hrv_calculation = time.time()
connected_device = None
is_device_connected = False

def classify_stress(hrv_metrics):
    """
    Classify stress level based on LF/HF ratio, RMSSD, and SDNN.
    """
    lf_hf = hrv_metrics.get("LF/HF Ratio", 0)
    rmssd = hrv_metrics.get("RMSSD (ms)", 0)
    sdnn = hrv_metrics.get("SDNN (ms)", 0)

    print(f"Debug - Stress Classification Inputs:")
    print(f"LF/HF Ratio: {lf_hf}")
    print(f"RMSSD: {rmssd}")
    print(f"SDNN: {sdnn}")

    # Heuristic rules based on literature
    if lf_hf > 8 or rmssd < 10 or sdnn < 20:
        stress_level = "High Stress"
    elif lf_hf > 4 or rmssd < 20 or sdnn < 40:
        stress_level = "Moderate Stress"
    else:
        stress_level = "Low Stress"
    
    print(f"Debug - Calculated Stress Level: {stress_level}")
    return stress_level

def compute_hrv_full(heart_rates):
    # Return zero values if no data or insufficient data
    zero_metrics = {
        "Mean RR (ms)": 0,
        "SDNN (ms)": 0,
        "RMSSD (ms)": 0,
        "Mean HR (bpm)": 0,
        "NN50 count": 0,
        "pNN50 (%)": 0,
        "AVNN (ms)": 0,
        "SDANN (ms)": 0,
        "SDNN index (ms)": 0,
        "TINN": 0,
        "HRV Triangular Index": 0,
        "VLF Power (ms²)": 0,
        "LF Power (ms²)": 0,
        "HF Power (ms²)": 0,
        "LF/HF Ratio": 0,
        "Total Power (ms²)": 0,
        "LFnu (%)": 0,
        "HFnu (%)": 0,
        "Stress Level": "No Data"
    }
    
    if len(heart_rates) < 120:
        print("Debug - Insufficient heart rate data points")
        return zero_metrics
        
    # Heart rates: [(timestamp, bpm), ...]
    rr_intervals = [60000 / hr for _, hr in heart_rates if hr > 0]
    
    if len(rr_intervals) < 120:
        print("Debug - Insufficient RR intervals")
        return zero_metrics

    # Time-domain features
    rr_diff = np.diff(rr_intervals)
    nn50 = np.sum(np.abs(rr_diff) > 50)
    pnn50 = nn50 / len(rr_diff) * 100 if len(rr_diff) > 0 else 0

    time_domain = {
        "Mean RR (ms)": float(np.mean(rr_intervals)),
        "SDNN (ms)": float(np.std(rr_intervals, ddof=1)),
        "RMSSD (ms)": float(np.sqrt(np.mean(rr_diff ** 2))),
        "NN50 count": int(nn50),
        "pNN50 (%)": float(pnn50),
    }

    # Additional time-domain features
    avnn = np.mean(rr_intervals)

    # SDANN
    segment_length = 100
    rr_intervals_segmented = [rr_intervals[i:i+segment_length] for i in range(0, len(rr_intervals), segment_length)]
    avg_rr_intervals_per_segment = [np.mean(segment) for segment in rr_intervals_segmented]
    sdann = np.std(avg_rr_intervals_per_segment) if len(avg_rr_intervals_per_segment) > 1 else 0

    # SDNN index
    sdnn_index = np.mean([np.std(segment, ddof=1) for segment in rr_intervals_segmented]) if rr_intervals_segmented else 0

    # TINN
    hist, bin_edges = np.histogram(rr_intervals, bins=50, density=False)
    
    #tinn = np.trapz(hist, bin_edges[1:] - bin_edges[:-1])
    bin_width = bin_edges[1] - bin_edges[0]
    tinn = np.trapz(hist, dx=bin_width)

    # HRV Triangular Index
    #hrv_triangular_index = len(rr_intervals) / np.max(hist) if len(rr_intervals) > 0 else 0
    hrv_triangular_index = len(rr_intervals) / np.max(hist)

    time_domain.update({
        "AVNN (ms)": float(avnn),
        "SDANN (ms)": float(sdann),
        "SDNN index (ms)": float(sdnn_index),
        "TINN": float(tinn),
        "HRV Triangular Index": float(hrv_triangular_index)
    })

    # Frequency-domain analysis
    try:
        rr_timestamps = np.cumsum(rr_intervals) / 1000.0
        #interpolator = interp1d(rr_timestamps, rr_intervals, kind='cubic', fill_value="extrapolate")
        interpolator = interp1d(rr_timestamps, rr_intervals, kind='linear', fill_value="extrapolate")
        time_uniform = np.linspace(rr_timestamps[0], rr_timestamps[-1], len(rr_timestamps))
        rr_interp = interpolator(time_uniform)

        freqs, psd = welch(rr_interp, fs=4.0)

        vlf_band = (0.003, 0.04)
        lf_band = (0.04, 0.15)
        hf_band = (0.15, 0.4)

        def band_power(freqs, psd, band):
            mask = (freqs >= band[0]) & (freqs < band[1])
            return np.trapz(psd[mask], freqs[mask])

        vlf = band_power(freqs, psd, vlf_band)
        lf = band_power(freqs, psd, lf_band)
        hf = band_power(freqs, psd, hf_band)
        lf_hf_ratio = lf / hf if hf > 0 else np.nan

        total_power = np.trapz(psd, freqs)
        lfnu = (lf / (total_power-vlf)) * 100 if (total_power-vlf) > 0 else 0
        hfnu = (hf / (total_power-vlf)) * 100 if (total_power-vlf) > 0 else 0

        freq_domain = {
            "VLF Power (ms²)": float(vlf),
            "LF Power (ms²)": float(lf),
            "HF Power (ms²)": float(hf),
            "LF/HF Ratio": float(lf_hf_ratio),
            "LFnu (%)": float(lfnu),
            "HFnu (%)": float(hfnu),
            "Total Power (ms²)": float(total_power),
            "Power Spectrum": {
                "frequencies": freqs.tolist(),
                "powers": psd.tolist()
            }
        }
    except Exception as e:
        print(f"Error in frequency domain calculation: {e}")
        freq_domain = {}

    # Combine all metrics
    all_metrics = {**time_domain, **freq_domain}
    
    # Add stress level classification
    stress_level = classify_stress(all_metrics)
    all_metrics["Stress Level"] = stress_level
    print(f"Debug - Final metrics stress level: {all_metrics['Stress Level']}")
    
    return all_metrics

def handle_hr_data(sender, data):
    global heart_rate_data, last_hrv_calculation, is_device_connected
    
    try:
        hr_value = data[1]
        current_time = datetime.now()
        
        # Add new heart rate data point
        heart_rate_data.append((current_time, hr_value))
        print("Appended to heart_rate_data:", (current_time, hr_value))
        print("Current heart_rate_data length:", len(heart_rate_data))
        print("Current heart_rate_data (last 5):", heart_rate_data[-5:])
        
        # Keep only last 2 minutes of data
        heart_rate_data = [x for x in heart_rate_data if (current_time - x[0]).total_seconds() <= 120]
        
        # Emit heart rate update
        socketio.emit('hr_update', {
            'heart_rate': hr_value,
            'timestamp': current_time.strftime('%H:%M:%S')
        })
        
        # Calculate HRV metrics every 2 minutes
        if time.time() - last_hrv_calculation >= 120:
            print("Emitting HRV metrics at:", datetime.now())
            hrv_metrics = compute_hrv_full(heart_rate_data)
            socketio.emit('hrv_update', hrv_metrics)  # Always emit, even if zero values
            last_hrv_calculation = time.time()
        
    except Exception as e:
        print(f"Error in handle_hr_data: {e}")
        socketio.emit('error', {'message': f'Error processing heart rate data: {str(e)}'})

def reset_metrics():
    """Reset all metrics to zero values and emit them to the frontend"""
    global is_device_connected
    is_device_connected = False
    
    zero_metrics = {
        "Mean RR (ms)": 0,
        "SDNN (ms)": 0,
        "RMSSD (ms)": 0,
        "Mean HR (bpm)": 0,
        "NN50 count": 0,
        "pNN50 (%)": 0,
        "AVNN (ms)": 0,
        "SDANN (ms)": 0,
        "SDNN index (ms)": 0,
        "TINN": 0,
        "HRV Triangular Index": 0,
        "VLF Power (ms²)": 0,
        "LF Power (ms²)": 0,
        "HF Power (ms²)": 0,
        "LF/HF Ratio": 0,
        "Total Power (ms²)": 0,
        "LFnu (%)": 0,
        "HFnu (%)": 0,
        "Stress Level": "No Data"
    }
    
    # Emit zero heart rate
    socketio.emit('hr_update', {
        'heart_rate': 0,
        'timestamp': datetime.now().strftime('%H:%M:%S')
    })
    
    # Emit zero HRV metrics
    socketio.emit('hrv_update', zero_metrics)
    
    # Emit disconnected status
    socketio.emit('status_update', {
        'status': 'disconnected',
        'message': 'No device connected'
    })
    
    # Emit device disconnected event
    socketio.emit('device_connected', {'name': None})

async def scan_and_connect():
    global connected_device, heart_rate_data, last_hrv_calculation, is_device_connected
    
    print("Scanning for BLE devices advertising HR data...")
    socketio.emit('status_update', {'status': 'scanning', 'message': 'Scanning for devices...'})
    
    while True:
        try:
            devices = await BleakScanner.discover()
            found_hr_devices = False

            for d in devices:
                if d.name:
                    print(f"Found device: {d.name} ({d.address})")
                    socketio.emit('status_update', {
                        'status': 'found_device',
                        'message': f'Found device: {d.name}'
                    })
                    
                    try:
                        async with BleakClient(d.address) as client:
                            services = await client.get_services()
                            if HR_UUID in [char.uuid for service in services for char in service.characteristics]:
                                print(f"{d.name} supports Heart Rate service!")
                                socketio.emit('status_update', {
                                    'status': 'connecting',
                                    'message': f'Connecting to {d.name}...'
                                })
                                
                                # Reset data when connecting to a new device
                                heart_rate_data = []
                                last_hrv_calculation = time.time()
                                
                                # Emit initial zero values
                                reset_metrics()
                                
                                connected_device = d.name
                                is_device_connected = True
                                socketio.emit('device_connected', {'name': d.name})
                                
                                await client.start_notify(HR_UUID, handle_hr_data)
                                print(f"Connected to {d.name} and receiving data...")
                                socketio.emit('status_update', {
                                    'status': 'connected',
                                    'message': f'Connected to {d.name} and receiving data'
                                })
                                
                                while True:
                                    await asyncio.sleep(1)
                    except Exception as e:
                        print(f"Failed to connect to {d.name}: {e}")
                        socketio.emit('status_update', {
                            'status': 'error',
                            'message': f'Failed to connect to {d.name}: {str(e)}'
                        })
                        # Reset metrics when connection fails
                        reset_metrics()
                        continue

            if not found_hr_devices:
                print("No heart rate devices found. Retrying in 5 seconds...")
                socketio.emit('status_update', {
                    'status': 'no_devices',
                    'message': 'No heart rate devices found. Retrying in 5 seconds...'
                })
                # Reset metrics when no devices are found
                reset_metrics()
                await asyncio.sleep(5)
        except Exception as e:
            print(f"Error in scan_and_connect: {e}")
            socketio.emit('status_update', {
                'status': 'error',
                'message': f'Error scanning for devices: {str(e)}'
            })
            # Reset metrics when there's an error
            reset_metrics()
            await asyncio.sleep(5)

def run_ble_scan():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(scan_and_connect())

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    print("Starting Heart Rate Monitor...")
    print("Please open http://127.0.0.1:5000")
    
    # Start BLE scanning in a separate thread
    ble_thread = threading.Thread(target=run_ble_scan)
    ble_thread.daemon = True
    ble_thread.start()
    
    # Start Flask server
    socketio.run(app, debug=True, use_reloader=False, allow_unsafe_werkzeug=True) 
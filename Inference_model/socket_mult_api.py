import websocket
import json
import threading

def on_accelerometer_event(values,timestamp):
    print(f"acclerometer values = {values}  timestamp = {timestamp}")

def on_gyroscope_event(values,timestamp):
    print(f"gyroscope values = {values}  timestamp = {timestamp}")

def on_magnetic_field_event(values,timestamp):
    print(f"magnetic field values = {values}  timestamp = {timestamp}")


class Sensor:
    
    def __init__(self,address,sensor_type, on_sensor_event):
        self.address = address
        self.sensor_type = sensor_type
        self.on_sensor_event = on_sensor_event
    
    def on_message(self,ws, message):
        values = json.loads(message)['values']
        timestamp = json.loads(message)['timestamp']

        self.on_sensor_event(values = values, timestamp = timestamp)

    def on_error(self,ws, error):
        print("error occurred")
        print(error)

    def on_close(self,ws, close_code, reason):
        print(f"connection closed : {reason}")

    def on_open(self,ws):
        print(f"connected to : {self.address}")

    def make_websocket_connection(self):
        ws = websocket.WebSocketApp(f"ws://{self.address}/sensor/connect?type={self.sensor_type}",
                                on_open=self.on_open,
                                on_message=self.on_message,
                                on_error=self.on_error,
                                on_close=self.on_close)

        # blocking call
        ws.run_forever()
    
    # make connection and start recieving data on sperate thread
    def connect(self):
        thread = threading.Thread(target=self.make_websocket_connection)
        thread.start() 

address = "192.168.88.53:8080"

Sensor(address = address, sensor_type="android.sensor.accelerometer", on_sensor_event=on_accelerometer_event).connect()
Sensor(address = address, sensor_type="android.sensor.gyroscope",on_sensor_event=on_gyroscope_event).connect()
Sensor(address = address, sensor_type="android.sensor.magnetic_field",on_sensor_event=on_magnetic_field_event).connect()



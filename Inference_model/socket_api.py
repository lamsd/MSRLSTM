import websocket
import json


def on_message(ws, message):
    values = json.loads(message)['values']
    print(values)
    # x = values[0]
    # y = values[1]
    # z = values[2]
    # print("x = ", x , "y = ", y , "z = ", z )

def on_error(ws, error):
    print("error occurred ", error)
    
def on_close(ws, close_code, reason):
    print("connection closed : ", reason)
    
def on_open(ws):
    print("connected")
    

def connect(url):
    ws = websocket.WebSocketApp(url,
                              on_open=on_open,
                              on_message=on_message,
                              on_error=on_error,
                              on_close=on_close)

    ws.run_forever()
 
 
# connect("ws://192.168.1.5:8080/sensors/connect?types=android.sensor.accelerometer") 

connect("ws://192.168.88.52:8080/sensors/connect?types=%5B%22android.sensor.gyroscope%22%2C%22android.sensor.accelerometer%22%5D")


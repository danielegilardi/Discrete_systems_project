from ctypes import alignment
import cv2
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import threading
from kafka import KafkaConsumer
import json
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc
from dash import html
import dash
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import paho.mqtt.client as mqtt
import os

class Visualizer:

    def __init__(self):
        
        self.body_parts = {}
        self.colormap = []
        cm_sections = np.linspace(0, 1, 61)
        cm = [ plt.cm.jet_r(x)[0:3] for x in cm_sections ]
        for i in cm:
            self.colormap.append(tuple(k * 255 for k in i))
        
        # coordinates on interest point, size of circle of color, time left before collision, last update of time left before collision
        self.body_parts[str(["nose","nose"])] = {'coord':[190, 95], 'size':30, 'col':False, 'lup':0, 'ct':0}
        self.body_parts[str(["left_wrist","left_wrist"])] = {'coord':[330, 455], 'size':30, 'col':False, 'lup':0, 'ct':0}
        self.body_parts[str(["right_wrist","right_wrist"])] = {'coord':[55, 455], 'size':30, 'col':False, 'lup':0, 'ct':0}
        self.body_parts[str(["nose","neck"])] = {'coord':[190, 140], 'size':30, 'col':False, 'lup':0, 'ct':0}
        self.body_parts[str(["left_shoulder","right_shoulder"])] = {'coord':[190, 185], 'size':30, 'col':False, 'lup':0, 'ct':0}
        self.body_parts[str(["left_shoulder","left_elbow"])] = {'coord':[290, 280], 'size':30, 'col':False, 'lup':0, 'ct':0}
        self.body_parts[str(["left_elbow","left_wrist"])] = {'coord':[310, 395], 'size':30, 'col':False, 'lup':0, 'ct':0}
        self.body_parts[str(["right_shoulder","right_elbow"])] = {'coord':[90, 280], 'size':30, 'col':False, 'lup':0, 'ct':0}
        self.body_parts[str(["right_elbow","right_wrist"])] = {'coord':[65, 395], 'size':30, 'col':False, 'lup':0, 'ct':0}
        self.body_parts[str(["right_shoulder","right_hip"])] = {'coord':[140, 350], 'size':30, 'col':False, 'lup':0, 'ct':0}
        self.body_parts[str(["left_shoulder","left_hip"])] = {'coord':[240, 350], 'size':30, 'col':False, 'lup':0, 'ct':0}
        self.body_parts[str(["left_hip","right_hip"])] = {'coord':[190, 425], 'size':30, 'col':False, 'lup':0, 'ct':0}
        self.body_parts[str(["right_hip","right_knee"])] = {'coord':[140, 540], 'size':40, 'col':False, 'lup':0, 'ct':0}
        self.body_parts[str(["right_knee","right_ankle"])] = {'coord':[135, 720], 'size':50, 'col':False, 'lup':0, 'ct':0}
        self.body_parts[str(["left_hip","left_knee"])] = {'coord':[240, 540], 'size':40, 'col':False, 'lup':0, 'ct':0}
        self.body_parts[str(["left_knee","left_ankle"])] = {'coord':[245, 720], 'size':50, 'col':False, 'lup':0, 'ct':0}
        self.body_parts[str(["chest","mid_hip"])] = {'coord':[190, 360], 'size':30, 'col':False, 'lup':0, 'ct':0}
        self.body_parts['rfoot'] = {'coord':[135, 850], 'size':50, 'col':False, 'lup':0, 'ct':0}
        self.body_parts['lfoot'] = {'coord':[245, 850], 'size':50, 'col':False, 'lup':0, 'ct':0}

        self.super_imposed_img = cv2.imread('./human.png')
        ex_col = self.super_imposed_img[0, 0]
        # Y and X are the y and x coordinates of background pixels
        self.Ybg, self.Xbg = np.where(np.all(self.super_imposed_img==ex_col, axis=-1))

    def print_body_parts(self):
        for key in self.body_parts.keys():
            print(key)

    def update_collision_times(self, key, t, ct):
        """
        Args:
            key : the string identifying the body part
            t : the time left before collision (in milliseconds)
            ct : currentTime of the message received
        """
        try:
            if ct > self.body_parts[key]['ct'] or t/1000 < self.body_parts[key]['col']:
                self.body_parts[key]['col'] = t/1000
                self.body_parts[key]['lup'] = time.time_ns()
                self.body_parts[key]['ct'] = ct
        except:
            print("{} == {} ? {}".format(str(["right_elbow","right_wrist"]), key, str(["right_elbow","right_wrist"])==key))
            print("Key not recognized!")

    def time_to_color(self, key):
        col = self.body_parts[key]['col']
        lup = self.body_parts[key]['lup']
        # if no collision time for the given segment is received for 500ms then the collision can be discarded
        if  col == False or time.time_ns() - lup >=  500000000 or col >= 55:
            return self.colormap[55]
        else:
            return self.colormap[int(col)]  

    def update_figure(self):
        """
        :return: updated "heatmap" of body parts (based on collision risk of each body part)
        """
        for key in self.body_parts.keys():
            self.super_imposed_img = cv2.circle(self.super_imposed_img,
                                                self.body_parts[key]['coord'],
                                                radius=self.body_parts[key]['size'],
                                                color=self.time_to_color(key),
                                                thickness=-1)

        self.super_imposed_img = cv2.blur(self.super_imposed_img, (50, 50))
        self.super_imposed_img[self.Ybg, self.Xbg] = [255, 255, 255]
        return self.super_imposed_img


def local_app(v : Visualizer):
    """
    This thread is for plotting the time left before collision
    """
    mpl.use("TkAgg")
    
    fig,[ax,cax] = plt.subplots(1,2, gridspec_kw={"width_ratios":[15,1]})
    cmap = mpl.cm.jet_r
    norm = mpl.colors.Normalize(vmin=0, vmax=55)
    cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                    norm=norm,
                                    orientation='vertical')
    cb1.ax.set_title('seconds')

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('Time before collision')
    im1 = ax.imshow(cv2.imread("./human.png"))

    def update(i):
        im1.set_data(v.update_figure())

    ani = FuncAnimation(fig, update, interval=100)
    plt.show()
    
    
def kafka_thread(v : Visualizer):
    """
    Kafka consumer to update the Visualizer collision times (with opera collision messages)
    """
    consumer = KafkaConsumer(
        'opera_data_collision_prediction', # topic name
        bootstrap_servers=os.environ['KAFKA_BROKER_URI'], # broker
        auto_offset_reset='latest',
        group_id="%s-consumer" % os.environ['KAFKA_USERNAME'],
        sasl_mechanism=os.environ['KAFKA_SASL_MECHANISM'],
        security_protocol=os.environ['KAFKA_SECURITY_PROTOCOL'],
        sasl_plain_username=os.environ['KAFKA_USERNAME'],
        sasl_plain_password=os.environ['KAFKA_PASSWORD'])
    
    print("Kafka consumer connected!")
    
    for msg in consumer:
        my_json = json.loads(msg.value)
        print(my_json)
        v.update_collision_times(str(my_json['human']['segment']), my_json['collisionDistance']['lower'], my_json['currentTime'])


def mqtt_thread(v : Visualizer, broker):
    
    def on_connect(client, userdata, flags, rc):
        print("Connected with result code " + str(rc))
        client.subscribe("opera_data_collision_prediction")

    def on_message(client, userdata, msg):
        my_json = json.loads(msg.payload.decode('utf8'))
        v.update_collision_times(str(my_json['human']['segment']), my_json['collisionDistance']['lower'], my_json['currentTime'])
        
        
    def on_disconnect(client, userdata, rc):
        pass 
    
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect

    client.connect(broker, 1883, 60)

    client.loop_forever()


def web_app(v : Visualizer):
    
    colorbar_trace=go.Scatter(x=[None],
             y=[None],
             mode='markers',
             marker=dict(
                 colorscale='Jet_r', 
                 showscale=True,
                 cmin=0,
                 cmax=60,
                 colorbar=dict(title='seconds', thickness=80, 
                               tickvals=[0, 10, 20, 30, 40, 50, 60], ticktext=['<=0', '10', '20', '30', '40', '50', '>=60'],
                               tickfont=dict(family='arial', size=20)
                               ), 
             ),
             hoverinfo='none'
            )
    layout_colorbar = dict(xaxis=dict(visible=False), yaxis=dict(visible=False), width=300, height=800, 
                           paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    colorbar = go.Figure([colorbar_trace], layout=layout_colorbar)
    colorbar.update_layout(coloraxis_colorbar_x=-0.1)
    
    # layout = go.Layout(autosize=False, width=600, height=800, margin=dict(l=50, r=50, b=100, t=100, pad=4))
    layout = go.Layout(autosize=False, width=700, height=800)
    fig = go.Figure(data=go.Image(z=v.update_figure()), layout=layout)
    fig.update_yaxes(visible=False, showticklabels=False)
    fig.update_xaxes(visible=False, showticklabels=False)
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    colors = {'background': '#FFFFFF'}
    
    app.layout = html.Div([
        dbc.Row(style={'height': '10vh'}),
        dbc.Row([
            dbc.Col(
                html.Div(
                            [
                                # dbc.Col(style={"height": "100%", "width": "25vw"}),
                                dbc.Col([html.H1(children='Time before collision',
                                         style={'textAlign': 'center', 'fontSize': '20'})], 
                                        style={"height": "100%", "width": "100vw"}),
                                # dbc.Col(style={"height": "100%", "width": "25vw"})
                            ],
                            style={"width": "100%", "height": "100%"}
                        )
            )],
        ),
        dbc.Row([
            dbc.Col(style={"height": "100%", "width": "5vw"}),
            dbc.Col(
                html.Div(
                    dcc.Graph(id="image",  figure=fig, config = {'staticPlot': True})
                ),
                style={"height": "100%", "width": "45vw"}
            ),
            dbc.Col(
                html.Div(
                    dcc.Graph(id="colorbar", figure=colorbar, config = {'staticPlot': True})
                    # style={"width": "100%", "height": "100%"}
                ),
                style={"height": "100%", "width": "50vw"}
            )],
        ),
        dcc.Interval(id="animateInterval", interval=1000)
    ], style={'backgroundColor': colors['background'], 'height': '95vh'}) # 'height': '100%', 'width': '100%', 

    @app.callback(
        Output("image", "figure"),
        [Input("animateInterval", "n_intervals")]
    )
    def doUpdate(n):
        updated = go.Figure(data=go.Image(z=v.update_figure()), layout=layout)
        updated.update_yaxes(visible=False, showticklabels=False)
        updated.update_xaxes(visible=False, showticklabels=False)
        return updated

    app.run(debug=False)


if __name__ == "__main__":
    
    v = Visualizer()
    
    # Create threads and pass arguments to target function
    thread = threading.Thread(target=kafka_thread, args=(v, ), daemon=True) 
    # thread = threading.Thread(target=mqtt_thread, args=(v, 'localhost'), daemon=True) 
    
    # Start threads
    thread.start()
    
    web_app(v)
    # local_app(v)
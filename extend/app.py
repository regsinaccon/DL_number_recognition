import tkinter as tk
import numpy
import os
from PIL import Image

def save_drawing(canvas, filename):
    canvas.postscript(file = filename, colormode='color')
    

root = tk.Tk()
root.title("Drawing App")
# 448 = 28*16
canvas = tk.Canvas(root, width=500, height=500)
canvas.pack()

# Create a Scale widget to change the thickness of the pen
pen_thickness = tk.Scale(root, from_=5, to=10, orient='horizontal')
pen_thickness.pack()

def start_draw(event):
    global lastx, lasty
    canvas.bind('<B1-Motion>', draw)
    lastx, lasty = event.x, event.y

def end_draw(event):
    canvas.unbind('<B1-Motion>')

def draw(event):
    global lastx, lasty
    x, y = event.x, event.y
    # Use the value of the Scale widget as the width of the line
    canvas.create_line((lastx, lasty, x, y), width=pen_thickness.get())
    lastx, lasty = x, y

# Function to clear the canvas
def clear_canvas():
    canvas.delete('all')

canvas.bind('<1>', start_draw)
canvas.bind('<ButtonRelease-1>', end_draw)

save_button = tk.Button(root, text="Save", command=lambda: save_drawing(canvas, "drawing.ps"))
save_button.place(x=20, y=480) 

# Add a Clear button that clears the canvas
clear_button = tk.Button(root, text="Clear", command=clear_canvas)
clear_button.place(x=400, y=480)  


save_button.config(height=2, width=10)
clear_button.config(height=2, width=10,)

root.mainloop()

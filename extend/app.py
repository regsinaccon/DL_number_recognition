import tkinter as tk

def save_drawing(canvas, filename):
    canvas.postscript(file = filename, colormode='color')

root = tk.Tk()
root.title("Drawing App")

canvas = tk.Canvas(root, width=28, height=28)
canvas.pack()

def start_draw(event):
    global lastx, lasty
    canvas.bind('<B1-Motion>', draw)
    lastx, lasty = event.x, event.y

def end_draw(event):
    canvas.unbind('<B1-Motion>')

def draw(event):
    global lastx, lasty
    x, y = event.x, event.y
    canvas.create_line((lastx, lasty, x, y))
    lastx, lasty = x, y

canvas.bind('<1>', start_draw)
canvas.bind('<ButtonRelease-1>', end_draw)

save_button = tk.Button(root, text="Save", command=lambda: save_drawing(canvas, "drawing.ps"))
save_button.pack()

root.mainloop()

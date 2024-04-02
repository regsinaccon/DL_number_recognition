from tkinter import *
import PIL
from PIL import Image, ImageDraw
from tkinter import messagebox
import recon

# Create a new Tk window instance
window = Tk()

# Create a new canvas for drawing
canvas = Canvas(window, width=448, height=460)
canvas.pack()

# Create an image for drawing
image = PIL.Image.new("L", (448, 448),255)
draw = ImageDraw.Draw(image)

def save_image():
    # Save the image as a PNG file
    image.save("drawing.png")
    num = recon.Pridict_Num()
    messagebox.showinfo("Result",f'The number is {num}')

def clear_image():
    global image, draw
    # Clear the canvas
    canvas.delete("all")
    # Create a new image for drawing
    image = PIL.Image.new("L", (448, 448),255)
    draw = ImageDraw.Draw(image)

last_pos = None

def draw_pixel(event):
    global last_pos
    # Draw a line from the last mouse position to the current one
    x, y = event.x, event.y
    if last_pos is not None:
        canvas.create_line(last_pos[0], last_pos[1], x, y, fill="black", width=25)
        draw.line([last_pos[0], last_pos[1], x, y], fill="black",width=25)
    last_pos = (x, y)

def reset_last_pos(event):
    global last_pos
    last_pos = None

# Bind the left mouse button down event to the draw_pixel function
canvas.bind("<B1-Motion>", draw_pixel)

# Bind the left mouse button up event to the reset_last_pos function
canvas.bind("<ButtonRelease-1>", reset_last_pos)

# Create a new button for saving the image
button_save = Button(window, text="Submit", command=save_image, height = 2, width = 10)
button_save.pack(side = "left")

# Create a new button for clearing the image
button_clear = Button(window, text="Clear", command=clear_image, height = 2, width = 10)
button_clear.pack(side = "right")

# Run the Tk main loop
window.mainloop()

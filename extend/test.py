import tkinter as tk

class PaintApp:
    def __init__(self, root):
        self.root = root
        self.canvas_width = 800
        self.canvas_height = 600
        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg="white", bd=3, relief=tk.SUNKEN)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.setup_navbar()
        self.setup_tools()
        self.setup_events()
        self.prev_x = None
        self.prev_y = None

    def setup_navbar(self):
        self.navbar = tk.Menu(self.root)

    def setup_tools(self):
        pass

    def setup_events(self):
        pass

# Initialize the main window
window = tk.Tk()
window.title("Paint App")

# Create the PaintApp
app = PaintApp(window)

# Start the main loop
window.mainloop()

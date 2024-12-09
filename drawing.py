from PIL import Image
import tkinter as tk
from tkinter import Canvas
import numpy as np
import os
import tempfile


class DrawingWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Draw Digit")

        scale = 15
        self.canvas_width = 28 * scale
        self.canvas_height = 28 * scale

        # Create canvas with black background
        self.canvas = Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg='black')
        self.canvas.pack(pady=20)

        self.drawing = False
        self.temp_path = None

        # Bind mouse events
        self.canvas.bind('<Button-1>', self.start_drawing)
        self.canvas.bind('<ButtonRelease-1>', self.stop_drawing)
        self.canvas.bind('<B1-Motion>', self.draw)

        # Add buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=5)

        clear_button = tk.Button(button_frame, text="Clear", command=self.clear_canvas)
        clear_button.pack(side=tk.LEFT, padx=5)

        save_button = tk.Button(button_frame, text="Save & Process", command=self.save_and_exit)
        save_button.pack(side=tk.LEFT, padx=5)

        # Add instructions
        instructions = tk.Label(self.root, text="Draw a digit using your mouse\nDraw in white, similar to MNIST digits",
                                justify=tk.CENTER)
        instructions.pack(pady=5)

    def start_drawing(self, event):
        self.drawing = True
        self.draw(event)  # Start drawing immediately

    def stop_drawing(self, event):
        self.drawing = False

    def draw(self, event):
        if self.drawing:
            x, y = event.x, event.y
            # Increased pen size for clearer drawing
            pen_size = 15
            self.canvas.create_oval(x - pen_size, y - pen_size, x + pen_size, y + pen_size,
                                    fill='white', outline='white')

    def clear_canvas(self):
        self.canvas.delete("all")
        self.canvas.configure(bg='black')

    def save_and_exit(self):
        # Create a temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        self.temp_path = temp_file.name
        temp_file.close()

        # Get canvas bounds
        x = self.canvas.winfo_rootx()
        y = self.canvas.winfo_rooty()
        x1 = x + self.canvas_width
        y1 = y + self.canvas_height

        # Create an image from canvas contents
        image = Image.new('L', (self.canvas_width, self.canvas_height), 'black')
        for item in self.canvas.find_all():
            coords = self.canvas.coords(item)
            if len(coords) >= 4:  # Only handle oval/rectangle items
                box = [int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])]
                # Draw white pixels for this item
                for i in range(box[0], box[2]):
                    for j in range(box[1], box[3]):
                        if 0 <= i < self.canvas_width and 0 <= j < self.canvas_height:
                            image.putpixel((i, j), 255)

        # Resize to MNIST dimensions
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        image.save(self.temp_path)

        self.root.destroy()
        return self.temp_path


def get_drawn_digit():
    """
    Opens drawing window and returns processed image using existing preprocessing
    """
    try:
        # Create and run drawing window
        window = DrawingWindow()
        window.root.mainloop()

        if window.temp_path is None:
            raise ValueError("No drawing was saved")

        # Use the existing preprocess_uploaded_image function
        from main import preprocess_uploaded_image
        processed_image = preprocess_uploaded_image(window.temp_path)

        # Clean up
        os.remove(window.temp_path)

        return processed_image

    except Exception as e:
        raise ValueError(f"Error processing drawing: {str(e)}")
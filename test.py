import tkinter as tk
from tkinter import ttk
from tkinter import Frame, Label
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk

def upload_image(panel):
    panel.destroy()

    file_path = filedialog.askopenfilename()
    image = Image.open(file_path)
    photo_image = ImageTk.PhotoImage(image)
    image_label = Label(panel, image=photo_image)
    image_label.pack()


# Create the main application window
root = tk.Tk()
root.title("Adversarial Attacks")
width= root.winfo_screenwidth()
height= root.winfo_screenheight()
root.geometry("%dx%d" % (width, height))

# Configure grid layout for the root window
root.rowconfigure(0, weight=1, minsize=150)  # Make top-left panel smaller
root.rowconfigure(1, weight=1)
root.columnconfigure(0, weight=1, minsize=150)  # Make top-left panel smaller
root.columnconfigure(1, weight=2)

# Function to create a panel with an inner box
def create_panel(parent, row, column, rowspan=1, columnspan=1):
    outer_frame = Frame(parent, bg="gray", bd=2, relief="sunken")
    outer_frame.grid(row=row, column=column, rowspan=rowspan, columnspan=columnspan, sticky="nsew", padx=5, pady=5)

    inner_frame = Frame(outer_frame, bg="gray", bd=2, relief="ridge")
    inner_frame.pack(fill="both", expand=True, padx=10, pady=10)

    return inner_frame

digit_options = ['MNIST Digit', 'Draw Digit', 'Upload Image', 'Camera Capture']

# Create top-left panel
top_left_panel = create_panel(root, row=0, column=0)

# Configure the top_left_panel for centering
top_left_panel.grid_columnconfigure(0, weight=1)  # Center align items horizontally
top_left_panel.grid_rowconfigure(tuple(range(len(digit_options))), weight=1)  # Distribute space equally

# # Create buttons in the top-left panel and center them
# for i, option in enumerate(digit_options):
#     button = tk.Button(top_left_panel, text=option, font=("Arial", 20, "bold"), command=lambda: print(option))
#     button.grid(row=i, column=0, pady=2, padx=2, sticky="nsew")  # Expands and centers

# Create buttons in the top-left panel and center them
mnist_btn = tk.Button(top_left_panel, text='MNIST Digit', font=("Arial", 20, "bold"))
mnist_btn.grid(row=0, column=0, pady=2, padx=2, sticky="nsew")

draw_btn = tk.Button(top_left_panel, text='Draw Digit', font=("Arial", 20, "bold"))
draw_btn.grid(row=1, column=0, pady=2, padx=2, sticky="nsew")

upload_btn = tk.Button(top_left_panel, text='Upload Image', font=("Arial", 20, "bold"), command=lambda: upload_image(top_left_panel))
upload_btn.grid(row=2, column=0, pady=2, padx=2, sticky="nsew")

camera_btn = tk.Button(top_left_panel, text='Camera Capture', font=("Arial", 20, "bold"))
camera_btn.grid(row=3, column=0, pady=2, padx=2, sticky="nsew")

# Create bottom-left panel
bottom_left_panel = create_panel(root, row=1, column=0)
# Create right panel
right_panel = create_panel(root, row=0, column=1, rowspan=2)

valid_digits = ['', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

target_class = tk.StringVar()
target_class.set(valid_digits[1])

target_label = ttk.Label(bottom_left_panel, text="Target Class:")
target_label.pack(pady=10, padx=10)
target_dropdown = ttk.OptionMenu(bottom_left_panel, target_class, *valid_digits)
target_dropdown.pack(pady=5, padx=5)

epsilon_label = ttk.Label(bottom_left_panel, text="Epsilon:")
epsilon_label.pack(pady=20)
epsilon_slider = tk.Scale(bottom_left_panel, from_=0.1, to=1.0, resolution=0.01, orient=HORIZONTAL, length=200)
epsilon_slider.pack()


# Run the Tkinter main event loop
root.mainloop()

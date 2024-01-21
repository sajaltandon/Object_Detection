import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions
import numpy as np

# Load pre-trained MobileNet model
model = MobileNet(weights='imagenet')

# Function to preprocess the input image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to make predictions on the input image
def predict_image(model, image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions)
    return decoded_predictions[0]

# Function to handle the "Identify" button click event
def identify_image():
    # Change the placeholder with your image path or keep it as is to select an image using the file dialog
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

    if file_path:
        # Display the selected image in the GUI
        img = Image.open(file_path)
        img = img.resize((500, 500))
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img

        # Make predictions and display the result in the Text widget
        predictions = predict_image(model, file_path)
        result_str = "Top Predictions:\n"
        for i, (imagenet_id, label, score) in enumerate(predictions):
            result_str += f"{i + 1}: {label} ({score:.2f})\n"

        result_text.config(state=tk.NORMAL)
        result_text.delete('1.0', tk.END)  # Clear previous results
        result_text.insert(tk.END, result_str)
        result_text.config(state=tk.DISABLED)

# Create the main window
root = tk.Tk()
root.title("Image Identifier")
root.geometry("600x700")  # Set the window size

# Styling
root.option_add('*Font', 'Helvetica 12')
root.option_add('*Button.Background', '#4CAF50')  # Green background color for the button
root.option_add('*Button.Foreground', 'white')    # White text color for the button

# Create and place GUI components
welcome_label = tk.Label(root, text="Welcome to Image Identifier!", font=('Helvetica', 18, 'bold'), pady=20)
welcome_label.pack()

upload_button = tk.Button(root, text="Upload Image", command=identify_image)
upload_button.pack(pady=20)

image_label = tk.Label(root)
image_label.pack()

# ScrolledText widget with a vertical scrollbar for identification results
result_text = scrolledtext.ScrolledText(root, height=8, width=50, state=tk.DISABLED, wrap=tk.WORD)
result_text.pack()

# Run the Tkinter event loop
root.mainloop()

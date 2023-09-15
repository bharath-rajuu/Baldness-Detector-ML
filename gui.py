import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model("baldness_model_final.h5")

# Create a function to make predictions and display the image


def predict_image():
    # Open a file dialog to select an image
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        # Resize the image to match the model's input size
        img = img.resize((224, 224))
        img = np.array(img) / 255.0  # Normalize pixel values

        # Make the prediction
        prediction = model.predict(np.expand_dims(img, axis=0))[0]

        if prediction > 0.5:
            result_label.config(text="Not Bald")
        else:
            result_label.config(text="Bald")

        # Display the image
        img = Image.open(file_path)
        img = img.resize((200, 200))  # Resize the image for display
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img  # Keep a reference to avoid garbage collection


# Create the main application window
app = tk.Tk()
app.title("Baldness Detection")

# Create a button to select an image
browse_button = tk.Button(app, text="Select Image", command=predict_image)
browse_button.pack(pady=20)

# Create a label to display the result
result_label = tk.Label(app, text="", font=("Helvetica", 18))
result_label.pack(pady=20)

# Create a label to display the image
image_label = tk.Label(app)
image_label.pack()

# Start the Tkinter main loop
app.mainloop()

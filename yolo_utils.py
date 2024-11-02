import colorsys
import imghdr
import os
import random
from keras import backend as K
import tensorflow as tf

import numpy as np
from PIL import Image, ImageDraw, ImageFont

def read_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def read_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
    return anchors

def generate_colors(class_names):
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors

def scale_boxes(boxes, image_shape):
    """Scales the predicted boxes in order to be drawable on the image."""
    height = image_shape[0]
    width = image_shape[1]

    # Use tf.stack instead of K.stack
    image_dims = tf.stack([height, width, height, width])
    image_dims = tf.reshape(image_dims, [1, 4])
    image_dims = tf.cast(image_dims, dtype=tf.float32)  # Cast to float32
    
    boxes = boxes * image_dims
    return boxes

def preprocess_image(img_path, model_image_size):
    image_type = imghdr.what(img_path)  # Check the image type (optional)
    image = Image.open(img_path)  # Open the image file
    resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)  # Resize the image
    image_data = np.array(resized_image, dtype='float32')  # Convert to numpy array
    image_data /= 255.  # Normalize the image data to the range [0, 1]
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension
    return image, resized_image, image_data  # Return original, resized, and normalized data


from PIL import ImageDraw, ImageFont
import numpy as np

from PIL import ImageDraw, ImageFont
import numpy as np

def draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors):
    # Load the default font
    font = ImageFont.load_default()
    thickness = (image.size[0] + image.size[1]) // 300

    # Initialize the drawing context
    draw = ImageDraw.Draw(image)

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)

        # Calculate label size using textbbox or fallback to textsize if not available
        try:
            label_bbox = draw.textbbox((0, 0), label, font=font)
            label_width = label_bbox[2] - label_bbox[0]
            label_height = label_bbox[3] - label_bbox[1]
        except AttributeError:
            # Fallback to textsize if textbbox is not available
            label_width, label_height = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

        # Ensure valid coordinates before drawing
        if left < right and top < bottom:
            print(label, (left, top), (right, bottom))

            # Calculate the position for the label
            if top - label_height >= 0:
                text_origin = np.array([left, top - label_height])
            else:
                text_origin = np.array([left, top + 1])

            # Draw the bounding box
            for j in range(thickness):
                draw.rectangle([left + j, top + j, right - j, bottom - j], outline=colors[c])

            # Draw the label background and text
            draw.rectangle([tuple(text_origin), tuple(text_origin + (label_width, label_height))], fill=colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)

    # Clean up
    del draw



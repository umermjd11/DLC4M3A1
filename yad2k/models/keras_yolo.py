"""YOLO_v2 Model Defined in Keras."""
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Concatenate, Layer, Lambda
from keras.models import Model

from ..utils import compose
from .keras_darknet19 import (DarknetConv2D, DarknetConv2D_BN_Leaky, darknet_body)

sys.path.append('..')

voc_anchors = np.array(
    [[1.08, 1.19], [3.42, 4.41], [6.63, 11.38], [9.42, 5.11], [16.62, 10.52]])

voc_classes = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


def space_to_depth_x2(x):
    """Thin wrapper for Tensorflow space_to_depth with block_size=2."""
    # Import currently required to make Lambda work.
    # See: https://github.com/fchollet/keras/issues/5088#issuecomment-273851273
    return tf.nn.space_to_depth(x, block_size=2)


def space_to_depth_x2_output_shape(input_shape):
    """Determine space_to_depth output shape for block_size=2.

    Note: For Lambda with TensorFlow backend, output shape may not be needed.
    """
    return (input_shape[0], input_shape[1] // 2, input_shape[2] // 2, 4 *
            input_shape[3]) if input_shape[1] else (input_shape[0], None, None,
                                                    4 * input_shape[3])


def yolo_body(inputs, num_anchors, num_classes):
    """Create YOLO_V2 model CNN body in Keras."""
    darknet = Model(inputs, darknet_body()(inputs))
    conv20 = compose(
        DarknetConv2D_BN_Leaky(1024, (3, 3)),
        DarknetConv2D_BN_Leaky(1024, (3, 3)))(darknet.output)

    conv13 = darknet.layers[43].output
    conv21 = DarknetConv2D_BN_Leaky(64, (1, 1))(conv13)
    # TODO: Allow Keras Lambda to use func arguments for output_shape?
    conv21_reshaped = Lambda(
        space_to_depth_x2,
        output_shape=space_to_depth_x2_output_shape,
        name='space_to_depth')(conv21)

    x = concatenate([conv21_reshaped, conv20])
    x = DarknetConv2D_BN_Leaky(1024, (3, 3))(x)
    x = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1))(x)
    return Model(inputs, x)


def old_yolo_head(feats, anchors, num_classes):
    """Convert final layer features to bounding box parameters.

    Parameters
    ----------
    feats : tensor
        Final convolutional layer features.
    anchors : array-like
        Anchor box widths and heights.
    num_classes : int
        Number of target classes.

    Returns
    -------
    box_xy : tensor
        x, y box predictions adjusted by spatial location in conv layer.
    box_wh : tensor
        w, h box predictions adjusted by anchors and conv spatial resolution.
    box_conf : tensor
        Probability estimate for whether each box contains any object.
    box_class_pred : tensor
        Probability distribution estimate for each box over class labels.
    """
    num_anchors = len(anchors)
    
    # Reshape anchors to a tensor
    anchors_tensor = tf.convert_to_tensor(anchors, dtype=tf.float32)
    anchors_tensor = tf.reshape(anchors_tensor, [1, 1, 1, num_anchors, 2])
    
    # Dynamic implementation of conv dims for fully convolutional model.
    conv_dims = tf.shape(feats)[1:3]  # assuming channels last
    # Create height and width indices
    conv_height_index = tf.range(0, conv_dims[0])
    conv_width_index = tf.range(0, conv_dims[1])
    
    # Tile the height index
    conv_height_index = tf.tile(tf.expand_dims(conv_height_index, axis=1), [1, conv_dims[1]])
    conv_height_index = tf.reshape(conv_height_index, [-1])
    
    # Tile the width index
    conv_width_index = tf.tile(tf.expand_dims(conv_width_index, axis=0), [conv_dims[0], 1])
    conv_width_index = tf.reshape(conv_width_index, [-1])
    
    # Stack the indices and reshape
    conv_index = tf.stack([conv_height_index, conv_width_index], axis=-1)
    conv_index = tf.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
    
    # Reshape feats to [batch_size, conv_height, conv_width, num_anchors, box_params]
    feats = tf.reshape(feats, [-1, conv_dims[0], conv_dims[1], num_anchors, num_classes + 5])
    
    # Adjust box parameters
    box_confidence = tf.sigmoid(feats[..., 4:5])
    box_xy = tf.sigmoid(feats[..., :2])
    box_wh = tf.exp(feats[..., 2:4])
    box_class_probs = tf.nn.softmax(feats[..., 5:])
    
    # Adjust predictions to each spatial grid point and anchor size.
    # Assuming box_xy and conv_index are float tensors and conv_dims is an int tensor
    box_xy = (box_xy + tf.cast(conv_index, tf.float32)) / tf.cast(conv_dims, tf.float32)
    box_wh = box_wh * anchors_tensor / tf.cast(conv_dims, tf.float32)

    return box_confidence, box_xy, box_wh, box_class_probs

def yolo_head_copied(feats, anchors, num_classes):
    
    anchors_tensor = tf.reshape(tf.Variable(anchors, dtype='float32'), [1, 1, 1, 5, 2])  

    yolo_model_outputs = feats

    array = yolo_model_outputs.numpy()             #converting tensor to numpy object for easy computation
    new_array = np.resize(array, (1,19,19,5,85))   #resizing from (None,19,19,425) -> (None,19,19,5,80)
    yolo_outputs = tf.convert_to_tensor(new_array) #converting back to tensor from numpy object 

    box_xy = tf.sigmoid(yolo_outputs[..., :2])              # Applying sigmoid to first 2 columns for getting x an y s.t. 
    box_wh = tf.exp(yolo_outputs[..., 2:4])                 # Applying exponential to first 3 and 4  columns for getting w an h s.t. (w,h >= 0) 
    box_confidence = tf.sigmoid(yolo_outputs[..., 4:5])     # Applying sigmoid to first 5 columns for getting x an y s.t. (0 >= p <= 1)
    box_class_probs = tf.nn.softmax(yolo_outputs[..., 5:])  # Applying softmax to first 6 to 85 columns for getting x an y s.t. (0 >= c <= 1)


    
    #creating conv_index tensor for scaling the x and y ( initialy computed w.r.t. grid cells ) 
    #coodinates with respect to the whole image (19 x 19 grid cells)
    #
    #instance of conv_index :-
    #conv_index = [[[[0,0],[0,1],[0,2],.........,[0,18]],
    #            [[1,0],[1,1],[1,2],...........,[1,18]],
    #            .......................................
    #            .......................................
    #            [[18,0],[18,1],[18,2],........,[18,18]]]]

    array = np.arange(0,19,1,dtype='float32')
    array = np.tile(array,(19,1))
    array_t = np.transpose(array)
    array2 = np.stack((array,array_t), -1)
    array3 = np.expand_dims(array2, axis=2)
    conv_index = np.expand_dims(array3, axis=0)
    conv_index = tf.convert_to_tensor(conv_index)


    #creating conv_dims tensor for scaling down the x,y,w,h w.r.t. the whole image 
    #
    #conv_dims = tensor(shape=[1,1,1,1,2], [19.0,19.0])

    dims = np.array([19.0,19.0],dtype='float32') 
    dims = np.reshape(dims,(1,1,1,1,2))
    conv_dims = tf.convert_to_tensor(dims)

    # applying operations to get final box_xy and box_wh 
    box_xy = (box_xy + conv_index) / conv_dims
    box_wh = box_wh * anchors_tensor / conv_dims

    return box_confidence, box_xy, box_wh, box_class_probs


def yolo_head(feats, anchors, num_classes):
    # Get batch size and grid size from feature map
    batch_size = tf.shape(feats)[0]
    grid_size = tf.shape(feats)[1:3]  # (grid_height, grid_width)

    # Reshape anchors to (1, 1, 1, num_anchors, 2) for broadcasting
    anchors_tensor = tf.reshape(tf.constant(anchors, dtype='float32'), [1, 1, 1, len(anchors), 2])  

    # Reshape feats to separate the anchor boxes and class probabilities
    yolo_outputs = tf.reshape(feats, (batch_size, grid_size[0], grid_size[1], len(anchors), num_classes + 5))

    # Extract components from the reshaped output
    box_xy = tf.sigmoid(yolo_outputs[..., :2])               # Center (x, y) coordinates
    box_wh = tf.exp(yolo_outputs[..., 2:4]) * anchors_tensor # Width, height with anchor scaling
    box_confidence = tf.sigmoid(yolo_outputs[..., 4:5])      # Object confidence
    box_class_probs = tf.nn.softmax(yolo_outputs[..., 5:])   # Class probabilities

    # Generate grid offsets to add to `box_xy` for proper scaling
    grid_x = tf.range(grid_size[1], dtype=tf.float32)   # Width indices
    grid_y = tf.range(grid_size[0], dtype=tf.float32)   # Height indices
    grid_x, grid_y = tf.meshgrid(grid_x, grid_y)        # Create grid
    grid = tf.stack([grid_x, grid_y], axis=-1)          # Shape: (grid_height, grid_width, 2)
    grid = tf.expand_dims(grid, 2)                      # Shape: (grid_height, grid_width, 1, 2)
    grid = tf.expand_dims(grid, 0)                      # Shape: (1, grid_height, grid_width, 1, 2)
    grid = tf.tile(grid, [batch_size, 1, 1, len(anchors), 1])  # Tile for batch and anchors

    # Compute final box coordinates
    box_xy = (box_xy + grid) / tf.cast(grid_size[::-1], dtype=tf.float32)  # Normalize to 0-1
    box_wh = box_wh / tf.cast(grid_size[::-1], dtype=tf.float32)           # Normalize width/height

    return box_confidence, box_xy, box_wh, box_class_probs


import tensorflow as tf
from tensorflow.keras.layers import Layer


def yolo_boxes_to_corners(box_xy, box_wh):
    """Convert YOLO box predictions to bounding box corners."""
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    # Use tf.concat instead of K.concatenate
    return tf.concat([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ], axis=-1)


def yolo_loss(args,
              anchors,
              num_classes,
              rescore_confidence=False,
              print_loss=False):
    """YOLO localization loss function.

    Parameters
    ----------
    yolo_output : tensor
        Final convolutional layer features.

    true_boxes : tensor
        Ground truth boxes tensor with shape [batch, num_true_boxes, 5]
        containing box x_center, y_center, width, height, and class.

    detectors_mask : array
        0/1 mask for detector positions where there is a matching ground truth.

    matching_true_boxes : array
        Corresponding ground truth boxes for positive detector positions.
        Already adjusted for conv height and width.

    anchors : tensor
        Anchor boxes for model.

    num_classes : int
        Number of object classes.

    rescore_confidence : bool, default=False
        If true then set confidence target to IOU of best predicted box with
        the closest matching ground truth box.

    print_loss : bool, default=False
        If True then use a tf.Print() to print the loss components.

    Returns
    -------
    mean_loss : float
        mean localization loss across minibatch
    """
    (yolo_output, true_boxes, detectors_mask, matching_true_boxes) = args
    num_anchors = len(anchors)
    object_scale = 5
    no_object_scale = 1
    class_scale = 1
    coordinates_scale = 1
    pred_xy, pred_wh, pred_confidence, pred_class_prob = yolo_head(
        yolo_output, anchors, num_classes)

    # Unadjusted box predictions for loss.
    # TODO: Remove extra computation shared with yolo_head.
    yolo_output_shape = K.shape(yolo_output)
    feats = K.reshape(yolo_output, [
        -1, yolo_output_shape[1], yolo_output_shape[2], num_anchors,
        num_classes + 5
    ])
    pred_boxes = K.concatenate(
        (K.sigmoid(feats[..., 0:2]), feats[..., 2:4]), axis=-1)

    # TODO: Adjust predictions by image width/height for non-square images?
    # IOUs may be off due to different aspect ratio.

    # Expand pred x,y,w,h to allow comparison with ground truth.
    # batch, conv_height, conv_width, num_anchors, num_true_boxes, box_params
    pred_xy = K.expand_dims(pred_xy, 4)
    pred_wh = K.expand_dims(pred_wh, 4)

    pred_wh_half = pred_wh / 2.
    pred_mins = pred_xy - pred_wh_half
    pred_maxes = pred_xy + pred_wh_half

    true_boxes_shape = K.shape(true_boxes)

    # batch, conv_height, conv_width, num_anchors, num_true_boxes, box_params
    true_boxes = K.reshape(true_boxes, [
        true_boxes_shape[0], 1, 1, 1, true_boxes_shape[1], true_boxes_shape[2]
    ])
    true_xy = true_boxes[..., 0:2]
    true_wh = true_boxes[..., 2:4]

    # Find IOU of each predicted box with each ground truth box.
    true_wh_half = true_wh / 2.
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half

    intersect_mins = K.maximum(pred_mins, true_mins)
    intersect_maxes = K.minimum(pred_maxes, true_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas

    # Best IOUs for each location.
    best_ious = K.max(iou_scores, axis=4)  # Best IOU scores.
    best_ious = K.expand_dims(best_ious)

    # A detector has found an object if IOU > thresh for some true box.
    object_detections = K.cast(best_ious > 0.6, K.dtype(best_ious))

    # TODO: Darknet region training includes extra coordinate loss for early
    # training steps to encourage predictions to match anchor priors.

    # Determine confidence weights from object and no_object weights.
    # NOTE: YOLO does not use binary cross-entropy here.
    no_object_weights = (no_object_scale * (1 - object_detections) *
                         (1 - detectors_mask))
    no_objects_loss = no_object_weights * K.square(-pred_confidence)

    if rescore_confidence:
        objects_loss = (object_scale * detectors_mask *
                        K.square(best_ious - pred_confidence))
    else:
        objects_loss = (object_scale * detectors_mask *
                        K.square(1 - pred_confidence))
    confidence_loss = objects_loss + no_objects_loss

    # Classification loss for matching detections.
    # NOTE: YOLO does not use categorical cross-entropy loss here.
    matching_classes = K.cast(matching_true_boxes[..., 4], 'int32')
    matching_classes = K.one_hot(matching_classes, num_classes)
    classification_loss = (class_scale * detectors_mask *
                           K.square(matching_classes - pred_class_prob))

    # Coordinate loss for matching detection boxes.
    matching_boxes = matching_true_boxes[..., 0:4]
    coordinates_loss = (coordinates_scale * detectors_mask *
                        K.square(matching_boxes - pred_boxes))

    confidence_loss_sum = K.sum(confidence_loss)
    classification_loss_sum = K.sum(classification_loss)
    coordinates_loss_sum = K.sum(coordinates_loss)
    total_loss = 0.5 * (
        confidence_loss_sum + classification_loss_sum + coordinates_loss_sum)
    if print_loss:
        total_loss = tf.Print(
            total_loss, [
                total_loss, confidence_loss_sum, classification_loss_sum,
                coordinates_loss_sum
            ],
            message='yolo_loss, conf_loss, class_loss, box_coord_loss:')

    return total_loss


def yolo(inputs, anchors, num_classes):
    """Generate a complete YOLO_v2 localization model."""
    num_anchors = len(anchors)
    body = yolo_body(inputs, num_anchors, num_classes)
    outputs = yolo_head(body.output, anchors, num_classes)
    return outputs


def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=.6):
    """Filter YOLO boxes based on object and class confidence."""

    box_scores = box_confidence * box_class_probs
    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1)
    prediction_mask = box_class_scores >= threshold

    # TODO: Expose tf.boolean_mask to Keras backend?
    boxes = tf.boolean_mask(boxes, prediction_mask)
    scores = tf.boolean_mask(box_class_scores, prediction_mask)
    classes = tf.boolean_mask(box_classes, prediction_mask)

    return boxes, scores, classes




def yolo_eval(yolo_outputs,
              image_shape,
              max_boxes=10,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input batch and return filtered boxes."""
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
    boxes, scores, classes = yolo_filter_boxes(
        box_confidence, boxes, box_class_probs, threshold=score_threshold)
    
    # Scale boxes back to original image shape.
    height = image_shape[0]
    width = image_shape[1]
    image_dims = K.stack([height, width, height, width])
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims

    # TODO: Something must be done about this ugly hack!
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))
    nms_index = tf.image.non_max_suppression(
        boxes, scores, max_boxes_tensor, iou_threshold=iou_threshold)
    boxes = K.gather(boxes, nms_index)
    scores = K.gather(scores, nms_index)
    classes = K.gather(classes, nms_index)
    
    return boxes, scores, classes


def preprocess_true_boxes(true_boxes, anchors, image_size):
    """Find detector in YOLO where ground truth box should appear.

    Parameters
    ----------
    true_boxes : array
        List of ground truth boxes in form of relative x, y, w, h, class.
        Relative coordinates are in the range [0, 1] indicating a percentage
        of the original image dimensions.
    anchors : array
        List of anchors in form of w, h.
        Anchors are assumed to be in the range [0, conv_size] where conv_size
        is the spatial dimension of the final convolutional features.
    image_size : array-like
        List of image dimensions in form of h, w in pixels.

    Returns
    -------
    detectors_mask : array
        0/1 mask for detectors in [conv_height, conv_width, num_anchors, 1]
        that should be compared with a matching ground truth box.
    matching_true_boxes: array
        Same shape as detectors_mask with the corresponding ground truth box
        adjusted for comparison with predicted parameters at training time.
    """
    height, width = image_size
    num_anchors = len(anchors)
    # Downsampling factor of 5x 2-stride max_pools == 32.
    # TODO: Remove hardcoding of downscaling calculations.
    assert height % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'
    assert width % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'
    conv_height = height // 32
    conv_width = width // 32
    num_box_params = true_boxes.shape[1]
    detectors_mask = np.zeros(
        (conv_height, conv_width, num_anchors, 1), dtype=np.float32)
    matching_true_boxes = np.zeros(
        (conv_height, conv_width, num_anchors, num_box_params),
        dtype=np.float32)

    for box in true_boxes:
        # scale box to convolutional feature spatial dimensions
        box_class = box[4:5]
        box = box[0:4] * np.array(
            [conv_width, conv_height, conv_width, conv_height])
        i = np.floor(box[1]).astype('int')
        j = min(np.floor(box[0]).astype('int'),1)
        best_iou = 0
        best_anchor = 0
                
        for k, anchor in enumerate(anchors):
            # Find IOU between box shifted to origin and anchor box.
            box_maxes = box[2:4] / 2.
            box_mins = -box_maxes
            anchor_maxes = (anchor / 2.)
            anchor_mins = -anchor_maxes

            intersect_mins = np.maximum(box_mins, anchor_mins)
            intersect_maxes = np.minimum(box_maxes, anchor_maxes)
            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_area = intersect_wh[0] * intersect_wh[1]
            box_area = box[2] * box[3]
            anchor_area = anchor[0] * anchor[1]
            iou = intersect_area / (box_area + anchor_area - intersect_area)
            if iou > best_iou:
                best_iou = iou
                best_anchor = k
                
        if best_iou > 0:
            detectors_mask[i, j, best_anchor] = 1
            adjusted_box = np.array(
                [
                    box[0] - j, box[1] - i,
                    np.log(box[2] / anchors[best_anchor][0]),
                    np.log(box[3] / anchors[best_anchor][1]), box_class
                ],
                dtype=np.float32)
            matching_true_boxes[i, j, best_anchor] = adjusted_box
    return detectors_mask, matching_true_boxes

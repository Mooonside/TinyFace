import tensorflow as tf
import numpy as np

ANCHOR_SIZES = np.asarray(
  [[-62.5, -82.5, 62.5, 82.5],
   [-54.22727203, -62.54545593, 54.22727203, 62.54545593],
   [-50.5625, -42., 50.5625, 42.],
   [-46.66666794, -69.83333588, 46.66666794, 69.83333588],
   [-46.375, -76.125, 46.375, 76.125],
   [-45.95454407, -60.59090805, 45.95454407, 60.59090805],
   [-41.43103409, -52.89655304, 41.43103409, 52.89655304],
   [-37.23529434, -29.16742134, 37.23529434, 29.16742134],
   [-36.5633812, -39.09859085, 36.5633812, 39.09859085],
   [-36.3139534, -45.9186058, 36.3139534, 45.9186058],
   [-30.97752762, -40.53932571, 30.97752762, 40.53932571],
   [-30.43770981, -32.88215637, 30.43770981, 32.88215637],
   [-29.30078125, -19.54296875, 29.30078125, 19.54296875],
   [-28.53125, -25.92708397, 28.53125, 25.92708397],
   [-25.5357151, -38.17582321, 25.5357151, 38.17582321],
   [-24.14436531, -30.38908386, 24.14436531, 30.38908386],
   [-21.3844223, -24.86683464, 21.3844223, 24.86683464],
   [-20.16666603, -14.34615421, 20.16666603, 14.34615421],
   [-17.63074112, -20.76590157, 17.63074112, 20.76590157],
   [-16.45270348, -28.58108139, 16.45270348, 28.58108139],
   [-14.45134735, -17.74550819, 14.45134735, 17.74550819],
   [-11.22577286, -12.82783508, 11.22577286, 12.82783508],
   [-9.48584938, -21.98113251, 9.48584938, 21.98113251],
   [-7.58214283, -8.98928547, 7.58214283, 8.98928547],
   [-1.75, -1.5, 1.75, 1.5]],dtype=np.float32)

RGB_MEAN = [119.29959869,  110.54627228,  101.83843231]

def tf_ssd_bboxes_encode_layer(labels,
                               bboxes,
                               anchors_layer,
                               num_classes,
                               no_annotation_label,
                               ignore_threshold=0.5,
                               prior_scaling=[0.1, 0.1, 0.2, 0.2],
                               dtype=tf.float32):
  """Encode groundtruth labels and bounding boxes using SSD anchors from
  one layer.

  Arguments:
    labels: 1D Tensor(int64) containing groundtruth labels;
    bboxes: Nx4 Tensor(float) with bboxes relative coordinates;
    anchors_layer: Numpy array with layer anchors;
    matching_threshold: Threshold for positive match with groundtruth bboxes;
    prior_scaling: Scaling of encoded coordinates.

  Return:
    (target_labels, target_localizations, target_scores): Target Tensors.
  """
  # Anchors coordinates and volume.
  yref, xref, href, wref = anchors_layer
  ymin = yref - href / 2.
  xmin = xref - wref / 2.
  ymax = yref + href / 2.
  xmax = xref + wref / 2.
  vol_anchors = (xmax - xmin) * (ymax - ymin)

  # Initialize tensors...
  # how many default bboxes in total
  shape = (yref.shape[0], yref.shape[1], href.size)
  feat_labels = tf.zeros(shape, dtype=tf.int64)
  feat_scores = tf.zeros(shape, dtype=dtype)

  feat_ymin = tf.zeros(shape, dtype=dtype)
  feat_xmin = tf.zeros(shape, dtype=dtype)
  feat_ymax = tf.ones(shape, dtype=dtype)
  feat_xmax = tf.ones(shape, dtype=dtype)

  def jaccard_with_anchors(bbox):
    """Compute jaccard score between a box and the anchors.
    """
    int_ymin = tf.maximum(ymin, bbox[0])
    int_xmin = tf.maximum(xmin, bbox[1])
    int_ymax = tf.minimum(ymax, bbox[2])
    int_xmax = tf.minimum(xmax, bbox[3])
    h = tf.maximum(int_ymax - int_ymin, 0.)
    w = tf.maximum(int_xmax - int_xmin, 0.)
    # Volumes.
    inter_vol = h * w
    union_vol = vol_anchors - inter_vol \
                + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    jaccard = tf.div(inter_vol, union_vol)
    return jaccard

  def intersection_with_anchors(bbox):
    """Compute intersection between score a box and the anchors.
    """
    int_ymin = tf.maximum(ymin, bbox[0])
    int_xmin = tf.maximum(xmin, bbox[1])
    int_ymax = tf.minimum(ymax, bbox[2])
    int_xmax = tf.minimum(xmax, bbox[3])
    h = tf.maximum(int_ymax - int_ymin, 0.)
    w = tf.maximum(int_xmax - int_xmin, 0.)
    inter_vol = h * w
    scores = tf.div(inter_vol, vol_anchors)
    return scores

  def condition(i, feat_labels, feat_scores,
                feat_ymin, feat_xmin, feat_ymax, feat_xmax):
    """Condition: check label index.
    """
    r = tf.less(i, tf.shape(labels))
    return r[0]

  def body(i, feat_labels, feat_scores,
           feat_ymin, feat_xmin, feat_ymax, feat_xmax):
    """
      Body: update feature labels, scores and bboxes.
      Follow the original SSD paper for that purpose:
        - assign values when jaccard > 0.5;
        - only update if beat the score of other bboxes.
    """
    # Jaccard score.
    label = labels[i]
    bbox = bboxes[i]
    # calculate jaccard distance of gt_bboxes and anchor_bboxes
    jaccard = jaccard_with_anchors(bbox)
    # Mask: check threshold + scores + no annotations + num_classes.
    # remove those jaccard = 0
    mask = tf.greater(jaccard, feat_scores)
    # mask = tf.logical_and(mask, tf.greater(jaccard, matching_threshold))
    mask = tf.logical_and(mask, feat_scores > -0.5)
    mask = tf.logical_and(mask, label < num_classes)
    imask = tf.cast(mask, tf.int64)
    fmask = tf.cast(mask, dtype)
    # Update values using mask.
    feat_labels = imask * label + (1 - imask) * feat_labels
    feat_scores = tf.where(mask, jaccard, feat_scores)

    feat_ymin = fmask * bbox[0] + (1 - fmask) * feat_ymin
    feat_xmin = fmask * bbox[1] + (1 - fmask) * feat_xmin
    feat_ymax = fmask * bbox[2] + (1 - fmask) * feat_ymax
    feat_xmax = fmask * bbox[3] + (1 - fmask) * feat_xmax

    # Check no annotation label: ignore these anchors...
    # interscts = intersection_with_anchors(bbox)
    # mask = tf.logical_and(interscts > ignore_threshold,
    #                       label == no_annotation_label)
    # # Replace scores by -1.
    # feat_scores = tf.where(mask, -tf.cast(mask, dtype), feat_scores)

    return [i + 1, feat_labels, feat_scores,
            feat_ymin, feat_xmin, feat_ymax, feat_xmax]

  # Main loop definition.
  i = 0
  [i, feat_labels, feat_scores,
   feat_ymin, feat_xmin,
   feat_ymax, feat_xmax] = tf.while_loop(condition, body,
                                         [i, feat_labels, feat_scores,
                                          feat_ymin, feat_xmin,
                                          feat_ymax, feat_xmax])
  # Transform to center / size.
  feat_cy = (feat_ymax + feat_ymin) / 2.
  feat_cx = (feat_xmax + feat_xmin) / 2.
  feat_h = feat_ymax - feat_ymin
  feat_w = feat_xmax - feat_xmin
  # Encode features.
  feat_cy = (feat_cy - yref) / href / prior_scaling[0]
  feat_cx = (feat_cx - xref) / wref / prior_scaling[1]
  feat_h = tf.log(feat_h / href) / prior_scaling[2]
  feat_w = tf.log(feat_w / wref) / prior_scaling[3]
  # Use SSD ordering: x / y / w / h instead of ours.
  feat_localizations = tf.stack([feat_cx, feat_cy, feat_w, feat_h], axis=-1)
  return feat_labels, feat_localizations, feat_scores


def tinyface_bboxes_encode(labels, bboxes, anchors, num_classes,
                           scope='tinyface_bboxes_encode', dtype=tf.float32):
  with tf.name_scope(scope, 'tinyface_bboxes_encode'):
    anchors_x, anchors_y, anchors_width, anchors_height = anchors
    num_of_ground_truth = tf.shape(labels)

    # do broadcasting
    anchors_x = np.expand_dims(anchors_x, axis=-1)
    anchors_y = np.expand_dims(anchors_y, axis=-1)

    # has shape [f_h, f_w, #anchors]
    anchors_xmin = anchors_x - anchors_width / 2
    anchors_xmax = anchors_x + anchors_width / 2
    anchors_ymin = anchors_y - anchors_height / 2
    anchors_ymax = anchors_y + anchors_height / 2
    vol_anchors = (anchors_xmax - anchors_xmin) * (anchors_ymax - anchors_ymin)

    shape = anchors_xmin.shape
    anchors_scores = tf.zeros(shape=shape, dtype=dtype)
    anchors_labels = tf.zeros(shape=shape, dtype=tf.int64)
    feat_ymin = tf.zeros(shape, dtype=dtype)
    feat_xmin = tf.zeros(shape, dtype=dtype)
    feat_ymax = tf.ones(shape, dtype=dtype)
    feat_xmax = tf.ones(shape, dtype=dtype)

    # Iteratively assign ground truth bboxes to feat_xxx
    def jaccard_with_anchors(bbox):
      """Compute jaccard score between a box and the anchors.
      """
      int_ymin = tf.maximum(anchors_ymin, bbox[0])
      int_xmin = tf.maximum(anchors_xmin, bbox[1])
      int_ymax = tf.minimum(anchors_ymax, bbox[2])
      int_xmax = tf.minimum(anchors_xmax, bbox[3])
      h = tf.maximum(int_ymax - int_ymin, 0.)
      w = tf.maximum(int_xmax - int_xmin, 0.)
      # Volumes.
      inter_vol = h * w
      union_vol = vol_anchors - inter_vol \
                  + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
      jaccard = tf.div(inter_vol, union_vol)
      return jaccard

    def condition(i, feat_labels, feat_scores,
             feat_ymin, feat_xmin, feat_ymax, feat_xmax):
      """Condition: check label index.
      """
      r = tf.less(i, num_of_ground_truth)
      return r[0]

    def body(i, feat_labels, feat_scores,
             feat_ymin, feat_xmin, feat_ymax, feat_xmax):
      """
        Body: update feature labels, scores and bboxes.
        Follow the original SSD paper for that purpose:
          - assign values when jaccard > 0.5;
          - only update if beat the score of other bboxes.
      """
      # Jaccard score.
      label = labels[i]
      bbox = bboxes[i]
      # calculate jaccard distance of gt_bboxes and anchor_bboxes
      jaccard = jaccard_with_anchors(bbox)
      # Mask: check threshold + scores + no annotations + num_classes.
      # remove those jaccard = 0
      mask = tf.greater(jaccard, feat_scores)
      # mask = tf.logical_and(mask, tf.greater(jaccard, matching_threshold))
      mask = tf.logical_and(mask, feat_scores > -0.5)
      mask = tf.logical_and(mask, label < num_classes)
      imask = tf.cast(mask, tf.int64)
      fmask = tf.cast(mask, dtype)
      # Update values using mask.
      feat_labels = imask * label + (1 - imask) * feat_labels
      feat_scores = tf.where(mask, jaccard, feat_scores)

      feat_ymin = fmask * bbox[0] + (1 - fmask) * feat_ymin
      feat_xmin = fmask * bbox[1] + (1 - fmask) * feat_xmin
      feat_ymax = fmask * bbox[2] + (1 - fmask) * feat_ymax
      feat_xmax = fmask * bbox[3] + (1 - fmask) * feat_xmax

      # Check no annotation label: ignore these anchors...
      # interscts = intersection_with_anchors(bbox)
      # mask = tf.logical_and(interscts > ignore_threshold,
      #                       label == no_annotation_label)
      # # Replace scores by -1.
      # feat_scores = tf.where(mask, -tf.cast(mask, dtype), feat_scores)

      return [i + 1, feat_labels, feat_scores,
              feat_ymin, feat_xmin, feat_ymax, feat_xmax]
    # Main loop definition.
    i = 0
    [i, feat_labels, feat_scores,
     feat_ymin, feat_xmin,
     feat_ymax, feat_xmax] = tf.while_loop(condition, body,
                                           [i, anchors_labels, anchors_scores,
                                            feat_ymin, feat_xmin,
                                            feat_ymax, feat_xmax])

    # Transform to center - size style.
    feat_cy = (feat_ymax + feat_ymin) / 2.
    feat_cx = (feat_xmax + feat_xmin) / 2.
    feat_h = feat_ymax - feat_ymin
    feat_w = feat_xmax - feat_xmin
    # Encode features for bbox regression
    feat_cy = (feat_cy - anchors_y) / anchors_height
    feat_cx = (feat_cx - anchors_x) / anchors_width
    feat_h = tf.log(feat_h / anchors_height)
    feat_w = tf.log(feat_w / anchors_width)
    # Use SSD ordering: x / y / w / h instead of ours.
    feat_localizations = tf.stack([feat_cx, feat_cy, feat_w, feat_h], axis=-1)
    return feat_labels, feat_localizations, feat_scores


def tf_ssd_bboxes_encode(labels,
                         bboxes,
                         anchors,
                         num_classes,
                         no_annotation_label,
                         ignore_threshold=0.5,
                         prior_scaling=[0.1, 0.1, 0.2, 0.2],
                         dtype=tf.float32,
                         scope='ssd_bboxes_encode'):
  """Encode ground truth labels and bounding boxes using SSD net anchors.
  Encoding boxes for all feature layers.

  Arguments:
    labels: 1D Tensor(int64) containing groundtruth labels;
    bboxes: Nx4 Tensor(float) with bboxes relative coordinates;
    anchors: List of Numpy array with layer anchors;
    matching_threshold: Threshold for positive match with groundtruth bboxes;
    prior_scaling: Scaling of encoded coordinates.

  Return:
    (target_labels, target_localizations, target_scores):
      Each element is a list of target Tensors.
  """
  with tf.name_scope(scope):
    target_labels = []
    target_localizations = []
    target_scores = []
    for i, anchors_layer in enumerate(anchors):
      with tf.name_scope('bboxes_encode_block_%i' % i):
        t_labels, t_loc, t_scores = \
          tf_ssd_bboxes_encode_layer(labels, bboxes, anchors_layer,
                                     num_classes, no_annotation_label,
                                     ignore_threshold,
                                     prior_scaling, dtype)
        target_labels.append(t_labels)
        target_localizations.append(t_loc)
        target_scores.append(t_scores)
    return target_labels, target_localizations, target_scores
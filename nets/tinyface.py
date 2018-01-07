from collections import namedtuple
import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets import custom_layers
from nets import resnet_v1
from nets import tinyface_common
from .tinyface_utils import conv2d_trans
from .tinyface_common import ANCHOR_SIZES
import tf_extended as tfe
import numpy as np

TFParams = namedtuple('TFParameters', ['img_shape',
                                       'num_classes',
                                       'no_annotation_label',
                                       # 'feat_layer',
                                       'feat_shape',
                                       'anchor_sizes',
                                       # 'anchor_ratios',
                                       'anchor_steps',
                                       'anchor_offset',
                                       # 'normalizations',
                                       # 'prior_scaling'
                                       ])


class TinyFace:
  def __init__(self, params=None):

    # define default parameters
    default_params = TFParams(
      img_shape=(500, 500),
      num_classes=2,
      no_annotation_label=2,
      feat_shape=[32, 32],
      anchor_offset=0.5,
      anchor_steps=16,
      anchor_sizes=ANCHOR_SIZES
    )
    if isinstance(params, TFParams):
      self.params = params
    else:
      self.params = default_params
    self.input_shape = self.params.img_shape

  def TF_net(self, inputs, is_training):
    return tf_res50_net(inputs, self.params.num_classes, is_training=is_training,
                        scope="tf_res50_net")

  def losses(self, logits, localisations,
             gclasses, glocalisations, gscores,
             match_threshold=0.5,
             negative_ratio=3.,
             alpha=1.,
             label_smoothing=0.,
             scope='ssd_losses'):
    """Define the SSD network losses.
    """
    return tinyface_losses(logits, localisations,
                      gclasses, glocalisations, gscores,
                      match_threshold=match_threshold,
                      negative_ratio=negative_ratio,
                      alpha=alpha,
                      label_smoothing=label_smoothing,
                      scope=scope)

  def anchors(self):
    img_h, img_w = self.params.img_shape
    feat_h, feat_w = self.params.feat_shape
    offset = self.params.anchor_offset
    step = self.params.anchor_steps
    anchors = self.params.anchor_sizes
    feat_x, feat_y = np.mgrid[0:feat_w, 0:feat_h]

    # map feature scale to image scale
    anchors_x = (feat_x + offset) * step / img_w
    anchors_y = (feat_y + offset) * step / img_h

    anchors_width = (anchors[:, 2] - anchors[:, 0]) / img_w
    anchors_height = (anchors[:, 3] - anchors[:, 1]) / img_h

    anchors_x = anchors_x.astype(np.float32)
    anchors_y = anchors_y.astype(np.float32)
    anchors_width = anchors_width.astype(np.float32)
    anchors_height = anchors_height.astype(np.float32)

    return [anchors_x, anchors_y, anchors_width, anchors_height]

  def bboxes_encode(self, labels, bboxes, anchors,
                    ignore_threshold=0.5, scope=None):
    """
      Encode labels and bounding boxes.
    """
    return tinyface_common.tinyface_bboxes_encode(
      labels, bboxes, anchors,
      self.params.num_classes,
      scope=scope
    )


def tf_arg_scope(weight_decay=0.0005, data_format='NHWC'):
  """
  Defines tiny face arg scope.

  Args:
    weight_decay: The l2 regularization coefficient.

  Returns:
    An arg_scope.
  """
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      weights_initializer=tf.contrib.layers.xavier_initializer(),
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                        padding='SAME',
                        data_format=data_format):
      with slim.arg_scope([custom_layers.pad2d,
                           custom_layers.l2_normalization,
                           custom_layers.channel_to_last],
                          data_format=data_format) as sc:
        return sc


def tensor_shape(x, rank=3):
  """Returns the dimensions of a tensor.
  Args:
    x: A N-D Tensor of shape.
  Returns:
    A list of dimensions. Dimensions that are statically known are python
      integers,otherwise they are integer scalar tensors.
  """
  if x.get_shape().is_fully_defined():
    return x.get_shape().as_list()
  else:
    static_shape = x.get_shape().with_rank(rank).as_list()
    dynamic_shape = tf.unstack(tf.shape(x), rank)
    return [s if s is not None else d
            for s, d in zip(static_shape, dynamic_shape)]


def tf_res50_net(inputs, num_classes, is_training, scope='tf_res50_net'):
  resnet_arg_scope = resnet_v1.resnet_arg_scope()
  resnet_v1_50 = resnet_v1.resnet_v1_50

  with slim.arg_scope(resnet_arg_scope):
    # inputs = tf.placeholder(dtype=tf.float32, shape=[32, 224, 224, 3])
    end_points = resnet_v1_50(inputs, is_training=is_training, scope=scope)

  res2 = end_points[1][scope + '/block1']
  res3 = end_points[1][scope + '/block2']
  res4 = end_points[1][scope + '/block3']

  num_anchors = ANCHOR_SIZES.shape[0]
  num_scores = num_anchors * num_classes
  num_coords = num_anchors * 4
  num_features = num_scores + num_coords

  batch_size = res3.get_shape()[0].value
  H_res3 = res3.get_shape()[1].value
  W_res3 = res3.get_shape()[2].value

  dres2 = slim.conv2d(res2, num_features, 1, stride=2, padding='SAME', scope='down_res3')

  ures4 = conv2d_trans(res4, filter_shape=[1, 1, num_features, 1024], strides=[1, 2, 2, 1],
                       output_shape=[batch_size, H_res3, W_res3, num_features],
                       padding='SAME', scope='up_res4')

  res3 = slim.conv2d(res3, num_features, 1, stride=1, scope='res3')

  fpyramid = tf.add_n([res3, dres2, ures4], name='fpyramid')

  scores, localisations = tf.split(fpyramid, [num_scores, num_coords], axis=3)

  # solve for prediction
  # [bs, H, W, A, C]
  scores = tf.reshape(scores, tensor_shape(scores)[:-1] + [num_anchors, num_classes])
  # [bs, H, W, A, 4]
  localisations = tf.reshape(localisations, tensor_shape(localisations)[:-1] + [num_anchors, 4])

  predictions = slim.softmax(logits=scores)

  return predictions, localisations, scores, end_points


def tinyface_losses(logits, localisations,
                    gclasses, glocalisations, gscores,
                    match_threshold=0.5,
                    negative_ratio=3.,
                    alpha=1.,
                    label_smoothing=0.,
                    device='/cpu:0',
                    scope=None):
  with tf.name_scope(scope, 'tintface_losses'):
    lshape = tfe.get_shape(logits[0], 5)
    num_classes = lshape[-1]
    batch_size = lshape[0]

    # Flatten out all vectors!
    logits = tf.reshape(logits, [-1, num_classes])
    gclasses = tf.reshape(gclasses, [-1])
    gscores = tf.reshape(gscores, [-1])
    localisations = tf.reshape(localisations, [-1, 4])
    glocalisations = tf.reshape(glocalisations, [-1, 4])
    # And concat the crap!

    dtype = logits.dtype

    # Compute positive matching mask...
    pmask = gscores > match_threshold
    fpmask = tf.cast(pmask, dtype)
    n_positives = tf.reduce_sum(fpmask)

    # Hard negative mining...
    no_classes = tf.cast(pmask, tf.int32)
    predictions = slim.softmax(logits)
    # find those have [0, 0.5] ious
    nmask = tf.logical_and(tf.logical_not(pmask),
                           gscores > -0.5)
    fnmask = tf.cast(nmask, dtype)
    # for negative bboxes, assign their scores to nvalues. o.w. assign 1
    nvalues = tf.where(nmask,
                       predictions[:, 0],
                       1. - fnmask)
    nvalues_flat = tf.reshape(nvalues, [-1])
    # Number of negative entries to select.
    max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
    n_neg = tf.cast(negative_ratio * n_positives, tf.int32) + batch_size
    n_neg = tf.minimum(n_neg, max_neg_entries)
    # hard negative : choose those negative bboxes with lowest scores
    val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
    max_hard_pred = -val[-1]
    # Final negative mask.
    nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
    fnmask = tf.cast(nmask, dtype)
    total_loss = 0

    # Add cross-entropy loss.
    with tf.name_scope('cross_entropy_pos'):
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                            labels=gclasses)
      loss = tf.div(tf.reduce_sum(loss * fpmask), batch_size, name='value')
      total_loss += loss
      tf.losses.add_loss(loss)

    with tf.name_scope('cross_entropy_neg'):
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                            labels=no_classes)
      loss = tf.div(tf.reduce_sum(loss * fnmask), batch_size, name='value')
      total_loss += loss
      tf.losses.add_loss(loss)

    # Add localization loss: smooth L1, L2, ...
    with tf.name_scope('localization'):
      # Weights Tensor: positive mask + random negative.
      weights = tf.expand_dims(alpha * fpmask, axis=-1)
      loss = custom_layers.abs_smooth(localisations - glocalisations)
      loss = tf.div(tf.reduce_sum(loss * weights), batch_size, name='value')
      total_loss += loss
      tf.losses.add_loss(loss)
    return total_loss

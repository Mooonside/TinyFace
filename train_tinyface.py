import tensorflow as tf
import os

from datasets import dataset_factory
from preprocess import preprocessing_factory
from nets import nets_factory
import tf_utils
from tensorflow.python.ops import control_flow_ops

slim = tf.contrib.slim

PREFIX_PATH = '/media/yifeng/Yifeng_Seagate/DataSets'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

DATA_FORMAT = 'NHWC'

tf.app.flags.DEFINE_string(
  'train_dir', os.path.join(PREFIX_PATH, 'tmp/tf_models/'),
  'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer(
  'num_readers', 4,
  'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
  'num_preprocessing_threads', 4,
  'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
  'log_every_n_steps', 10,
  'The frequency with which logs are print.')
tf.app.flags.DEFINE_integer(
  'save_summaries_secs', 600,
  'The frequency with which summaries are saved, in seconds.')
tf.app.flags.DEFINE_integer(
  'save_interval_secs', 600,
  'The frequency with which the model is saved, in seconds.')
tf.app.flags.DEFINE_float(
  'gpu_memory_fraction', 0.8, 'GPU memory fraction to use.')


# =========================================================================== #
# Optimization Flags.
# =========================================================================== #
tf.app.flags.DEFINE_float(
  'weight_decay', 0.00004, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_string(
  'optimizer', 'rmsprop',
  'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
  '"ftrl", "momentum", "sgd" or "rmsprop".')
tf.app.flags.DEFINE_float(
  'adadelta_rho', 0.95,
  'The decay rate for adadelta.')
tf.app.flags.DEFINE_float(
  'adagrad_initial_accumulator_value', 0.1,
  'Starting value for the AdaGrad accumulators.')
tf.app.flags.DEFINE_float(
  'adam_beta1', 0.9,
  'The exponential decay rate for the 1st moment estimates.')
tf.app.flags.DEFINE_float(
  'adam_beta2', 0.999,
  'The exponential decay rate for the 2nd moment estimates.')
tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')
tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')
tf.app.flags.DEFINE_float(
  'ftrl_initial_accumulator_value', 0.1,
  'Starting value for the FTRL accumulators.')
tf.app.flags.DEFINE_float(
  'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')
tf.app.flags.DEFINE_float(
  'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')
tf.app.flags.DEFINE_float(
  'momentum', 0.9,
  'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

# =========================================================================== #
# Dataset Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
  'dataset_name', 'gree', 'The name of the dataset to load.')
tf.app.flags.DEFINE_integer(
  'num_classes', 2, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string(
  'dataset_split_name', 'train', 'The name of the train/test split.')
tf.app.flags.DEFINE_string(
  'dataset_dir', os.path.join(PREFIX_PATH, 'gree_dataset/tf_records'),
  'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_integer(
  'labels_offset', 0,
  'An offset for the labels in the dataset. This flag is primarily used to '
  'evaluate the VGG and ResNet architectures which do not use a background '
  'class for the ImageNet dataset.')
tf.app.flags.DEFINE_string(
  'model_name', 'tinyface', 'The name of the architecture to train.')
tf.app.flags.DEFINE_string(
  'preprocessing_name', 'gree', 'The name of the preprocessing to use. If left '
                                'as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_integer(
  'batch_size', 32, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer(
  'train_image_size', 1451 * 10, 'Train image size')
tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')

# =========================================================================== #
# TinyFace Network flags.
# =========================================================================== #
tf.app.flags.DEFINE_float(
  'loss_alpha', 1., 'Alpha parameter in the loss function.')
tf.app.flags.DEFINE_float(
  'negative_ratio', 3., 'Negative ratio in the loss function.')
tf.app.flags.DEFINE_float(
  'match_threshold', 0.5, 'Matching threshold in the loss function.')

# =========================================================================== #
# Learning Rate Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
  'learning_rate_decay_type',
  'exponential',
  'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
  ' or "polynomial"'
)
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
tf.app.flags.DEFINE_float(
  'end_learning_rate', 0.0001,
  'The minimal end learning rate used by a polynomial decay learning rate.')
tf.app.flags.DEFINE_float(
  'label_smoothing', 0.0, 'The amount of label smoothing.')
tf.app.flags.DEFINE_float(
  'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')
tf.app.flags.DEFINE_float(
  'num_epochs_per_decay', 2.0,
  'Number of epochs after which learning rate decays.')
tf.app.flags.DEFINE_float(
  'moving_average_decay', None,
  'The decay to use for the moving average.'
  'If left as None, then moving averages are not used.')

# =========================================================================== #
# Fine-Tuning Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
  'checkpoint_path', None,
  'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
  'checkpoint_model_scope', None,
  'Model scope in the checkpoint. None if the same as the trained model.')
tf.app.flags.DEFINE_string(
  'checkpoint_exclude_scopes', None,
  'Comma-separated list of scopes of variables to exclude when restoring '
  'from a checkpoint.')
tf.app.flags.DEFINE_string(
  'trainable_scopes', None,
  'Comma-separated list of scopes to filter the set of variables to train.'
  'By default, None would train all the variables.')
tf.app.flags.DEFINE_boolean(
  'ignore_missing_vars', False,
  'When restoring a checkpoint would ignore missing variables.')

FLAGS = tf.app.flags.FLAGS


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.DEBUG)
  with tf.Graph().as_default():
    global_step = slim.create_global_step()

    dataset = dataset_factory.get_dataset(
      FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
      preprocessing_name, is_training=True)

    provider = slim.dataset_data_provider.DatasetDataProvider(
      dataset,
      num_readers=FLAGS.num_readers,
      common_queue_capacity=20 * FLAGS.batch_size,
      common_queue_min=10 * FLAGS.batch_size,
      shuffle=True)

    tf_class = nets_factory.get_network(FLAGS.model_name)
    tf_model = tf_class()

    tf_anchors = tf_model.anchors()
    tf_input_shape = tf_model.input_shape
    [image, shape, glabels, gbboxes] = provider.get(['image', 'shape',
                                                     'object/label',
                                                     'object/bbox'])

    # Add Data Augmentation
    # image : -> random crop & resize & whitening
    # bboxes : -> random flip & change to relative style
    image, glabels, gbboxes = image_preprocessing_fn(image, glabels, gbboxes,
                                                     out_shape=tf_input_shape, data_format=DATA_FORMAT)

    # Match anchors to gt_bboxes
    gclasses, glocalisations, gscores = \
      tf_model.bboxes_encode(glabels, gbboxes, tf_anchors)

    b_image, b_gclasses, b_glocalisations, b_gscores = tf.train.batch(
      [image, gclasses, glocalisations, gscores],
      batch_size=FLAGS.batch_size,
      num_threads=FLAGS.num_preprocessing_threads,
      capacity=5 * FLAGS.batch_size)

    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    predictions, localisations, logits, end_points = \
      tf_model.TF_net(b_image, is_training=True)

    total_losses = tf_model.losses(logits, localisations,
                                   b_gclasses, b_glocalisations, b_gscores,
                                   match_threshold=FLAGS.match_threshold,
                                   negative_ratio=FLAGS.negative_ratio,
                                   alpha=FLAGS.loss_alpha,
                                   label_smoothing=FLAGS.label_smoothing)

    summaries.add(tf.summary.scalar('total_loss', total_losses))
    for loss in tf.get_collection(tf.GraphKeys.LOSSES):
      summaries.add(tf.summary.scalar(loss.op.name, loss))
    for loss in tf.get_collection('EXTRA_LOSSES'):
      summaries.add(tf.summary.scalar(loss.op.name, loss))
    for variable in slim.get_model_variables():
      summaries.add(tf.summary.histogram(variable.op.name, variable))

    if FLAGS.moving_average_decay:
      moving_average_variables = slim.get_model_variables()
      variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, global_step)
    else:
      moving_average_variables, variable_averages = None, None

    learning_rate = tf_utils.configure_learning_rate(FLAGS,
                                                     dataset.num_samples,
                                                     global_step)
    summaries.add(tf.summary.scalar('learning_rate', learning_rate))
    optimizer = tf_utils.configure_optimizer(FLAGS, learning_rate)
    # summaries.add(tf.summary.scalar('learning_rate', learning_rate))

    variables_to_train = tf_utils.get_variables_to_train(FLAGS)

    grad_vars = optimizer.compute_gradients(total_losses, variables_to_train)
    grad_updates = optimizer.apply_gradients(grad_vars,
                                             global_step=global_step)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if FLAGS.moving_average_decay:
      # Update ops executed locally by trainer.
      update_ops.append(variable_averages.apply(moving_average_variables))
    update_ops.append(grad_updates)
    update_op = tf.group(*update_ops)
    train_tensor = control_flow_ops.with_dependencies([update_op], total_losses,
                                                      name='train_op')

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
    config = tf.ConfigProto(log_device_placement=False,
                            gpu_options=gpu_options)
    saver = tf.train.Saver(max_to_keep=5,
                           keep_checkpoint_every_n_hours=1.0,
                           write_version=2,
                           pad_step_number=False)
    summary_op = tf.summary.merge(list(summaries), name='summary_op')


    debug = open('tf_variables.txt','w')
    for i in tf.global_variables():
      debug.write(i.name[:-2]+'\n')
    debug.close()

    slim.learning.train(
      train_tensor,
      logdir=FLAGS.train_dir,
      master='',
      is_chief=True,
      init_fn=tf_utils.get_init_fn(FLAGS),
      summary_op=summary_op,
      number_of_steps=FLAGS.max_number_of_steps,
      log_every_n_steps=FLAGS.log_every_n_steps,
      save_summaries_secs=FLAGS.save_summaries_secs,
      saver=saver,
      save_interval_secs=FLAGS.save_interval_secs,
      session_config=config,
      sync_optimizer=None)


if __name__ == '__main__':
  tf.app.run()

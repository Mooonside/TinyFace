
# coding: utf-8

# ## load dataset provider

# In[1]:


import tensorflow as tf
slim = tf.contrib.slim

from datasets import dataset_factory


class FLAGS(object):
    def __init__(self):
        return


FLAGS.num_readers = 4
FLAGS.batch_size=32
FLAGS.dataset_name = 'gree'
FLAGS.dataset_split_name = 'train'
FLAGS.dataset_dir = '/home/yifeng/Link\ to\ DataSets/gree_dataset/tf_records/'


dataset = dataset_factory.get_dataset(
      FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

provider = slim.dataset_data_provider.DatasetDataProvider(
  dataset,
  num_readers=FLAGS.num_readers,
  common_queue_capacity=20 * FLAGS.batch_size,
  common_queue_min=10 * FLAGS.batch_size,
  shuffle=True)


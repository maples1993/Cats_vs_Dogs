import tensorflow as tf
import numpy as np
import os


def get_all_files(file_path, is_random=True):
    """
    获取图片路径及其标签
    :param file_path: a sting, 图片所在目录
    :param is_random: True or False, 是否乱序
    :return:
    """
    image_list = []
    label_list = []

    cat_count = 0
    dog_count = 0
    for item in os.listdir(file_path):
        item_path = file_path + '\\' + item
        item_label = item.split('.')[0]  # 文件名形如  cat.0.jpg,只需要取第一个

        if os.path.isfile(item_path):
            image_list.append(item_path)
        else:
            raise ValueError('文件夹中有非文件项.')

        if item_label == 'cat':  # 猫标记为'0'
            label_list.append(0)
            cat_count += 1
        else:  # 狗标记为'1'
            label_list.append(1)
            dog_count += 1
    print('数据集中有%d只猫,%d只狗.' % (cat_count, dog_count))

    image_list = np.asarray(image_list)
    label_list = np.asarray(label_list)
    # 乱序文件
    if is_random:
        rnd_index = np.arange(len(image_list))
        np.random.shuffle(rnd_index)
        image_list = image_list[rnd_index]
        label_list = label_list[rnd_index]

    return image_list, label_list


def get_batch(train_list, image_size, batch_size, capacity, is_random=True):
    """
    获取训练批次
    :param train_list: 2-D list, [image_list, label_list]
    :param image_size: a int, 训练图像大小
    :param batch_size: a int, 每个批次包含的样本数量
    :param capacity: a int, 队列容量
    :param is_random: True or False, 是否乱序
    :return:
    """

    intput_queue = tf.train.slice_input_producer(train_list, shuffle=False)

    # 从路径中读取图片
    image_train = tf.read_file(intput_queue[0])
    image_train = tf.image.decode_jpeg(image_train, channels=3)  # 这里是jpg格式
    image_train = tf.image.resize_images(image_train, [image_size, image_size])
    image_train = tf.cast(image_train, tf.float32) / 255.  # 转换数据类型并归一化

    # 图片标签
    label_train = intput_queue[1]

    # 获取批次
    if is_random:
        image_train_batch, label_train_batch = tf.train.shuffle_batch([image_train, label_train],
                                                                      batch_size=batch_size,
                                                                      capacity=capacity,
                                                                      min_after_dequeue=100,
                                                                      num_threads=2)
    else:
        image_train_batch, label_train_batch = tf.train.batch([image_train, label_train],
                                                              batch_size=1,
                                                              capacity=capacity,
                                                              num_threads=1)
    return image_train_batch, label_train_batch


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # 测试图片读取
    image_dir = 'data\\train'
    train_list = get_all_files(image_dir, True)
    image_train_batch, label_train_batch = get_batch(train_list, 256, 1, 200, False)

    sess = tf.Session()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in range(10):
            if coord.should_stop():
                break

            image_batch, label_batch = sess.run([image_train_batch, label_train_batch])
            if label_batch[0] == 0:
                label = 'Cat'
            else:
                label = 'Dog'
            plt.imshow(image_batch[0]), plt.title(label)
            plt.show()

    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        coord.request_stop()

    coord.join(threads=threads)
    sess.close()
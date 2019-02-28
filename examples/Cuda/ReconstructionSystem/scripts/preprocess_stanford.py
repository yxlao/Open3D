import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description='generate data_association.txt per dataset')
    parser.add_argument(
        '--path', type=str, default='/home/wei/Work/data/stanford/')

    return parser.parse_args()


def generate_data_association(dataset_path):
    subdirs = os.listdir(dataset_path)

    depth_subdirs = ['depth']
    valid_depth_subdirs = [depth_subdir
                           for depth_subdir in depth_subdirs
                               if depth_subdir in subdirs]
    assert len(valid_depth_subdirs) == 1

    image_subdirs = ['color', 'rgb', 'image']
    valid_image_subdirs = [image_subdir
                           for image_subdir in image_subdirs
                               if image_subdir in subdirs]
    assert len(valid_image_subdirs) == 1

    depth_filenames = os.listdir(dataset_path + '/' + valid_depth_subdirs[0])
    sorted_depth_filenames = sorted(
        depth_filenames, key=lambda x: int(x.split('.')[0])
    )

    image_filenames = os.listdir(dataset_path + '/' + valid_image_subdirs[0])
    sorted_image_filenames = sorted(
        image_filenames, key=lambda x: int(x.split('.')[0].split('-')[0])
    )

    with open(dataset_path + '/data_association.txt', 'w') as fout:
        for depth_filename, image_filename in zip(
                sorted_depth_filenames, sorted_image_filenames):
            fout.write(valid_depth_subdirs[0] + '/' + depth_filename + ' ')
            fout.write(valid_image_subdirs[0] + '/' + image_filename)
            fout.write('\n')


if __name__ == '__main__':
    args = parse_args()
    base_path = args.path

    dataset_paths = os.listdir(base_path)
    for dataset_path in dataset_paths:
        dataset_absolute_path = base_path + '/' + dataset_path
        if os.path.isdir(dataset_absolute_path):
            print('Generating data associations for {}'.format(dataset_path))
            generate_data_association(dataset_absolute_path)

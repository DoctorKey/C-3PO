import os

def get_dataset_path(name):

    if name == 'PCD_raw':
        """
        ├── GSV
        │   ├── mask
        │   │   └── *.png
        │   ├── README.txt
        │   ├── t0
        │   │   └── *.jpg
        │   └── t1
        │       └── *.jpg
        └── TSUNAMI
            ├── mask
            │   └── *.png
            ├── README.txt
            ├── t0
            │   └── *.jpg
            └── t1
                └── *.jpg
        """
        # return '/data_path/pcd'
        return '/opt/Dataset/pcd/pcd'

    if name == 'PCD_CV':
        """
        ├── set0
        │   ├── test
        │   │   ├── mask
        │   │   ├── t0
        │   │   └── t1
        │   ├── test.txt
        │   ├── train
        │   │   ├── mask
        │   │   ├── t0
        │   │   └── t1
        │   └── train.txt
        ├── set1
        ├── set2
        ├── set3
        └── set4
        """
        # return '/data_path/pcd_5cv'
        return '/opt/Dataset/pcd/pcd_5cv'

    if name == 'CMU_binary':
        """
        ├── test
        │   ├── mask
        │   ├── t0
        │   └── t1
        └── train
            ├── mask
            ├── t0
            └── t1
        """
        # return '/data_path/VL-CMU-CD-binary255'
        return '/opt/Dataset/VL-CMU-CD/VL-CMU-CD-binary255'

    if name == 'CMU_raw':
        """
        ├── raw
        ├── test
        └── train
        """
        # return '/data_path/VL-CMU-CD'
        return '/opt/Dataset/VL-CMU-CD'

    if name == 'ChangeSim':
        """
        ├── dark_test_list.txt
        ├── dust_test_list.txt
        ├── idx2color.txt
        ├── Query
        │   ├── Query_Seq_Test
        │   └── Query_Seq_Train
        ├── Reference
        │   ├── Ref_Seq_Test
        │   └── Ref_Seq_Train
        ├── test_list.txt
        ├── test_split.txt
        ├── train_list.txt
        └── train_split.txt
        """
        return '/data_path/ChangeSim'

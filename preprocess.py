import numpy as np
import pandas as pd
import os
import random
from shutil import copyfile
import pydicom as dicom
import cv2
import json
from sklearn.model_selection import train_test_split

parent_dir = "chest_xray"
divide_dir = ["test", "val", "train",]
case_dir = ["COVID", "NORMAL", "PNEUMONIA"]

for i in range(0, len(divide_dir)):
    path_i = os.path.join(parent_dir, divide_dir[i])
    for j in range(0, len(case_dir)):
        path = os.path.join(path_i, case_dir[j])
        os.makedirs(path, exist_ok=True)

# this script is modified from https://github.com/lindawangg/COVID-Net

path_to_json = 'configs'

for file in os.listdir(path_to_json):

    # Opening JSON file
    fname = "%s/%s" % (path_to_json, file)
    f = open(fname, 'r')
    # Reading from file
    config = json.loads(f.read())
    print ('Reading json file: ', fname)

    csv =  pd.read_csv(config['csvpath'], usecols=config['extrac_cols'])

    if config['extrac_rows'] is not None:
        csv=csv.loc[(csv['view']==config['extrac_rows'])]

    # parameters for COVIDx dataset
    train = []
    test = []
    valid = []
    test_count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}
    train_count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}
    valid_count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}

    # avoid duplicates images
    patient_imgpath = {}

    # create a data structure that stores patient id, image filename, and finding
    filename_label = {'normal': [], 'pneumonia': [], 'COVID': []}
    count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}

    pneumonia_list = ["pneumonia", "SARS", "MERS", "Streptococcus", "Klebsiella", "Chlamydophila", "Legioella", "Lung Opacity", "1"]

    for index, row in csv.iterrows():
        f = str(row[config['result_col']]).split(',')[0] # take the first finding, for the case of COVID-19, ARDS
        if f.lower() == 'covid-19':
            count['COVID-19'] += 1
            if config['img_name_col'] is not None:
                entry = [str(row[config['id_col']]), row[config['img_name_col']], 'COVID-19']
            else:
                if os.path.exists(os.path.join(config['imgpath'], row[config['id_col']] + '.jpg')):
                    entry = [str(row[config['id_col']]), row[config['id_col']]+ '.jpg' , 'COVID-19']
                elif os.path.exists(os.path.join(config['imgpath'], row[config['id_col']] + '.png')):
                    entry = [str(row[config['id_col']]), row[config['id_col']]+ '.png',  'COVID-19']
                elif os.path.exists(os.path.join(config['imgpath'], row[config['id_col']] + '.dcm')):
                    entry = [str(row[config['id_col']]), row[config['id_col']]+ '.png',  'COVID-19']
            filename_label['COVID'].append(entry)
        elif f.lower() == 'no finding' or f.lower() == 'normal':
            count['normal'] += 1
            if config['img_name_col'] is not None:
                entry = [str(row[config['id_col']]), row[config['img_name_col']], 'normal']
            else:
                if os.path.exists(os.path.join(config['imgpath'], row[config['id_col']] + '.jpg')):
                    entry = [str(row[config['id_col']]), row[config['id_col']]+ '.jpg' , 'normal']
                elif os.path.exists(os.path.join(config['imgpath'], row[config['id_col']] + '.png')):
                    entry = [str(row[config['id_col']]), row[config['id_col']]+ '.png',  'normal']
                elif os.path.exists(os.path.join(config['imgpath'], row[config['id_col']] + '.dcm')):
                    entry = [str(row[config['id_col']]), row[config['id_col']]+ '.png',  'normal']
            filename_label['normal'].append(entry)
        elif f != 'nan' and any(f.lower() == p.lower() for p in pneumonia_list):
            count['pneumonia'] += 1
            if config['img_name_col'] is not None:
                entry = [str(row['patientid']), row['filename'], 'pneumonia']
            else:
                if os.path.exists(os.path.join(config['imgpath'], row[config['id_col']] + '.jpg')):
                    entry = [str(row[config['id_col']]), row[config['id_col']]+ '.jpg' , 'pneumonia']
                elif os.path.exists(os.path.join(config['imgpath'], row[config['id_col']] + '.png')):
                    entry = [str(row[config['id_col']]), row[config['id_col']]+ '.png', 'pneumonia']
                elif os.path.exists(os.path.join(config['imgpath'], row[config['id_col']] + '.dcm')):
                    entry = [str(row[config['id_col']]), row[config['id_col']]+ '.png',  'pneumonia']
            filename_label['pneumonia'].append(entry)

    print('Data distribution')

    patients_test= []
    patients_valid =[]
    patients_train = []

    # copy or write images to data directory
    for key in filename_label.keys():
        arr = np.array(filename_label[key])
        if arr.size == 0:
            continue
        elif arr.size < 10:
            # if available data is less than ten, import all data to traning set
            for patient in arr:
                if patient[0] not in patient_imgpath:
                    patient_imgpath[patient[0]] = [patient[1]]
                else:
                    if patient[1] not in patient_imgpath[patient[0]]:
                        patient_imgpath[patient[0]].append(patient[1])
                    else:
                        continue  # skip since image has already been written

                if config['img_format_dcm'] is True:
                    ds = dicom.dcmread(os.path.join(config['imgpath'], row[config['id_col']] + '.dcm'))
                    pixel_array_numpy = ds.pixel_array
                    imgname = patient + '.png'
                    cv2.imwrite(os.path.join(parent_dir, os.path.join(divide_dir[2], key), patient[1]), pixel_array_numpy)
                else:
                    copyfile(os.path.join(config['imgpath'], patient[1]), os.path.join(parent_dir, os.path.join(divide_dir[2], key), patient[1]))
                train.append(patient)
                train_count[patient[2]] += 1

        elif arr.size >= 10:
            # if available data is ten or more randomly split up patients by test, training, and validation sets
            patients_train, patients_test = train_test_split(arr[:, [0]], test_size=config['test_split'], random_state=42)
            patients_train, patients_valid = train_test_split(patients_train, test_size=config['valid_split'], random_state=42)
            #print('Key: ', key)
            #print('Test patients: ', patients_test.T)
            #print('Valid patients: ',patients_valid.T)

            for patient in arr:
                if patient[0] not in patient_imgpath:
                    patient_imgpath[patient[0]] = [patient[1]]
                else:
                    if patient[1] not in patient_imgpath[patient[0]]:
                        patient_imgpath[patient[0]].append(patient[1])
                    else:
                        continue  # skip since image has already been written

                if config['img_format_dcm'] is True:
                    ds = dicom.dcmread(os.path.join(config['imgpath'], row[config['id_col']] + '.dcm'))
                    pixel_array_numpy = ds.pixel_array
                    imgname = patient[0] + '.png'

                if patient[0] in patients_test:
                    if config['img_format_dcm'] is True:
                        cv2.imwrite(os.path.join(parent_dir, os.path.join(divide_dir[0], key), patient[1]), pixel_array_numpy)
                    else:
                        copyfile(os.path.join(config['imgpath'], patient[1]), os.path.join(parent_dir, os.path.join(divide_dir[0], key), patient[1]))
                    test.append(patient)
                    test_count[patient[2]] += 1
                elif patient[0] in patients_valid:
                    if config['img_format_dcm'] is True:
                        cv2.imwrite(os.path.join(parent_dir, os.path.join(divide_dir[1], key), patient[1]), pixel_array_numpy)
                    else:
                        copyfile(os.path.join(config['imgpath'], patient[1]), os.path.join(parent_dir, os.path.join(divide_dir[1], key), patient[1]))
                    valid.append(patient)
                    valid_count[patient[2]] += 1
                else:
                    if config['img_format_dcm'] is True:
                        cv2.imwrite(os.path.join(parent_dir, os.path.join(divide_dir[2], key), patient[1]), pixel_array_numpy)
                    else:
                        copyfile(os.path.join(config['imgpath'], patient[1]), os.path.join(parent_dir, os.path.join(divide_dir[2], key), patient[1]))
                    train.append(patient)
                    train_count[patient[2]] += 1

    print('test count: ', test_count)
    print('valid count: ', valid_count)
    print('train count: ', train_count)

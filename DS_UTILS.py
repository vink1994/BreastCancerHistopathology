import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from albumentations import *
from skimage.transform import resize
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score

class DDN_DS_PROCESS:
    def __init__(self, input_shape, batch_size, orders, proj_dir , dataset_mode, seed, train_test_ratio, augment=True):
        self.DDN_SHAPE                 = input_shape
        self.DDN_BATCHsize            = batch_size
        self.arr                   = orders
        self.DDN_DS_Variant          = dataset_mode
        self.DDN_NumSeed                  = seed
        self.TT_RATIO              = train_test_ratio
        self.AUG                   = augment
        self.proj_dir             = proj_dir
        self.GENERAL_CLASSES       = ["benign", "malignant"]
        self.BENIGN_SUB_CLASSES    = ["adenosis", "fibroadenoma", "phyllodes_tumor", "tubular_adenoma"]
        self.MALIGNANT_SUB_CLASSES = ["ductal_carcinoma", "lobular_carcinoma", "mucinous_carcinoma", "papillary_carcinoma"]
        
    def get_paths_n_labelsK(self):

        DDN_init_param1 = []
        DS_label = []

        for ix1, a in enumerate(self.GENERAL_CLASSES):
            if ix1 == 0:
                for ix2, b in enumerate(self.BENIGN_SUB_CLASSES):
                    path1 = self.proj_dir+a+"/SOB/"+b
                    for c in os.listdir(path1):
                        path2 = path1+"/"+c+"/"+self.DDN_DS_Variant
                        for img_name in os.listdir(path2):
                            path3 = path2+"/"+img_name
                            img_path = path3 
                            main_targets = np.zeros((2), dtype=np.float32) 
                            main_targets[ix1] = 1.
                            DDN_init_param1.append(img_path)
                            DS_label.append(main_targets)
                            break            
            if ix1 == 1:
                for ix2, b in enumerate(self.MALIGNANT_SUB_CLASSES):
                    path1 = self.proj_dir+a+"/SOB/"+b
                    for c in os.listdir(path1):
                        path2 = path1+"/"+c+"/"+self.DDN_DS_Variant
                        for img_name in os.listdir(path2):
                            path3 = path2+"/"+img_name
                            img_path = path3  
                            main_targets = np.zeros((2), dtype=np.float32) 
                            main_targets[ix1] = 1.
                            DDN_init_param1.append(img_path)
                            DS_label.append(main_targets)
                            break
                           
        return DDN_init_param1, DS_label
    
    def DDN_get_labelpath(self):

        DDN_init_param1      = []
        DS_label = []

        for ix1, a in enumerate(self.GENERAL_CLASSES):
            if ix1 == 0:
                for ix2, b in enumerate(self.BENIGN_SUB_CLASSES):
                    path1 = self.BASE_DIR+a+"/SOB/"+b
                    for c in os.listdir(path1):
                        path2 = path1+"/"+c+"/"+self.DDN_DS_Variant
                        for img_name in os.listdir(path2):
                            path3 = path2+"/"+img_name
                            img_path = path3 
                            main_targets = np.zeros((2), dtype=np.float32) 
                            main_targets[ix1] = 1.
                            DDN_init_param1.append(img_path)
                            DS_label.append(main_targets)

                            
            if ix1 == 1:
                for ix2, b in enumerate(self.MALIGNANT_SUB_CLASSES):
                    path1 = self.BASE_DIR+a+"/SOB/"+b
                    for c in os.listdir(path1):
                        path2 = path1+"/"+c+"/"+self.DDN_DS_Variant
                        for img_name in os.listdir(path2):
                            path3 = path2+"/"+img_name

                          
                            img_path = path3 

                            
                            main_targets = np.zeros((2), dtype=np.float32) 
                            main_targets[ix1] = 1.
                     
                           
                            DDN_init_param1.append(img_path)
                            DS_label.append(main_targets)
                           
        returnDDN_init_param1, DS_label
    
    def __len__(self):
        return len(self.DDN_get_labelpath()[0])
    
    def get_img(self, img_path):
        DS_IMg = Image.open(img_path)
        return np.array(DS_IMg)
    
    def DDN_proc_aug(self, DS_IMg):
        if self.AUG:
            augment = Compose([VerticalFlip(p=0.5),
                               HorizontalFlip(p=0.5),
                               RandomBrightnessContrast(p=0.3),
                               ShiftScaleRotate(p=0.5, shift_limit=0.2, scale_limit=0.2, rotate_limit=20)])  
        else:
            augment = Compose([])  

        DS_IMg = augment(image=DS_IMg)['image']
        return DS_IMg
    
    
    def DDN_Preproc(self, DS_IMg):
        DS_IMg = resize(DS_IMg, self.DDN_SHAPE)
        return DS_IMg
    
    def DDN_gen_shuff_DAT(self):
        img_paths, labels = self.DDN_get_labelpath()

        np.random.seed(self.DDN_NumSeed) 
        np.random.shuffle(img_paths)
        
        np.random.seed(self.DDN_NumSeed) 
        np.random.shuffle(labels)
        
        return img_paths, labels

    def get_ds_data(self):
        img_paths, labels = self.get_paths_n_labelsK()

       
        return img_paths, labels
        
    def Dataset_split_TT(self, get):  
        img_paths, labels = self.DDN_gen_shuff_DAT()
        x_train, x_test, y_train, y_test = train_test_split(img_paths, labels, test_size=self.TT_RATIO, random_state=self.DDN_NumSeed)
        
        if get=='train':
            return x_train, y_train
        
        elif get=='test':
            return x_test, y_test

    def split_train_test_ns(self, get):  
        img_paths, labels = self.DDN_get_labelpath()
        x_train, x_test, y_train, y_test = train_test_split(img_paths, labels, test_size=self.TT_RATIO, random_state=self.DDN_NumSeed,shuffle=False)
        
        if get=='train':
            return x_train, y_train
        
        elif get=='test':
            return x_test, y_test

    def get_pl_data(self, get):  
        img_paths, labels = self.get_ds_data()
        x_train = img_paths
        y_train = labels
        if get=='train':
            return x_train, y_train
        
        elif get=='test':
            return x_train, y_train
    
    def DDN_gen_DAT(self):
        img_paths, labels = self.Dataset_split_TT(get="train")
        
        while True:
            DDN_init_param1 = np.empty((self.DDN_BATCHsize,)+self.DDN_SHAPE, dtype=np.float32)
            DDN_init_param2 = np.empty((self.DDN_BATCHsize, 2), dtype=np.float32)

            batch = np.random.choice(self.arr, self.DDN_BATCHsize)

            for DDN_iter_param, id_ in enumerate(batch):
                img_path = img_paths[id_]
                DS_IMg = self.get_img(img_path)
                DS_IMg = self.DDN_proc_aug(DS_IMg)
                DS_IMg = self.DDN_Preproc(DS_IMg)
                DS_label = labels[id_]   
                DDN_init_param1[DDN_iter_param] = DS_IMg
                DDN_init_param2[DDN_iter_param] = DS_label

            yield DDN_init_param1,DDN_init_param2

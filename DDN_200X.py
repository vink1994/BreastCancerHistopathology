
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
import tensorflow.keras.backend as K 
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense,    \
                                    Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D, Input, Lambda, \
                                    LeakyReLU, MaxPooling2D, Multiply, Permute, Reshape, UpSampling2D   \

DDN_SHAPE = (224, 224, 3)
DDN_BATCHsize = 24
Model_epoch = 100
DDN_InitSplits = 5
DDN_NumSeed = 9
DDN_TTrat = 0.2

proj_dir    = "E:/mkfold/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/"
DDN_DS_Variant = "200X"
class DDN_DS_PROCESS:
    def __init__(self, input_shape, batch_size, orders, base_dir, dataset_mode, seed, train_test_ratio, augment=True):
        self.DDN_SHAPE                 = input_shape
        self.DDN_BATCHsize            = batch_size
        self.arr                   = orders
        self.DDN_DS_Variant          = dataset_mode
        self.DDN_NumSeed                  = seed
        self.TT_RATIO              = train_test_ratio
        self.AUG                   = augment
        
        self.proj_dir             = base_dir
        self.GENERAL_CLASSES       = ["benign", "malignant"]
        self.BENIGN_SUB_CLASSES    = ["adenosis", "fibroadenoma", "phyllodes_tumor", "tubular_adenoma"]
        self.MALIGNANT_SUB_CLASSES = ["ductal_carcinoma", "lobular_carcinoma", "mucinous_carcinoma", "papillary_carcinoma"]
        
        
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
        
    def Dataset_split_TT(self, get):  
        img_paths, labels = self.DDN_gen_shuff_DAT()
        x_train, x_test, y_train, y_test = train_test_split(img_paths, labels, test_size=self.TT_RATIO, random_state=self.DDN_NumSeed)
        
        if get=='train':
            return x_train, y_train
        
        elif get=='test':
            return x_test, y_test
    
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

            yieldDDN_init_param1,DDN_init_param2
dataset = DDN_DS_PROCESS(DDN_SHAPE, 1, range(4), BASE_DIR, DDN_DS_Variant, DDN_NumSeed, DDN_TTrat, augment=True)

for DDN_iter_param, data in enumerate(dataset.DDN_gen_DAT()):
    DS_IMg,DDN_init_param2 = data
    print(DS_IMg)
    print(DS_IMg.shape)
    print("-"*10)
    print(y)
    print(y.shape)
    print("-"*10)
    print(DS_IMg[0,:,:,:].shape)
    plt.imshow(DS_IMg[0,:,:,:])
    plt.show()
    
    if DDN_iter_param==0:
        break

def recall(y_true, y_pred):
   
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
  
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    precisionx = precision(y_true, y_pred)
    recallx = recall(y_true, y_pred)
    return 2*((precisionx*recallx)/(precisionx+recallx+K.epsilon()))

class SGDRScheduler(Callback):
    def __init__(self,
                 min_lr,
                 max_lr,
                 steps_per_epoch,
                 lr_decay=1,
                 cycle_length=10,
                 mult_factor=2):

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay

        self.batch_since_restart = 0
        self.next_restart = cycle_length

        self.steps_per_epoch = steps_per_epoch

        self.cycle_length = cycle_length
        self.mult_factor = mult_factor

        self.history = {}

    def DDN_comp_LR(self):
        
        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
        return lr

    def LR_Initialize(self, logs={}):
        
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.max_lr)

    def DDN_update_LR(self, batch, logs={}):
        
        logs = logs or {}
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.batch_since_restart += 1
        K.set_value(self.model.optimizer.lr, self.DDN_comp_LR())

    def DDN_checkend_cycle(self, epoch, logs={}):
       
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
            self.next_restart += self.cycle_length
            self.max_lr *= self.lr_decay
            self.best_weights = self.model.get_weights()

    def DDN_SetWeights(self, logs={}):
        
        self.model.set_weights(self.best_weights)

def DDN_PSNet_Block(psnet_Int_feature, ratio=8):
    
    psnet_Int_feature = channel_attention(psnet_Int_feature, ratio)
    psnet_Int_feature = DDN_CustomFeatSelect(psnet_Int_feature)
    return psnet_Int_feature

def channel_attention(input_feature, ratio=8):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]
    
    shared_layer_one = Dense(channel//ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    
    avg_pool = GlobalAveragePooling2D()(input_feature)    
    avg_pool = Reshape((1,1,channel))(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel//ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel)
    
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1,1,channel))(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1,1,channel//ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)
    
    psnet_Int_feature = Add()([avg_pool,max_pool])
    psnet_Int_feature = Activation('sigmoid')(psnet_Int_feature)

    if K.image_data_format() == "channels_first":
        psnet_Int_feature = Permute((3, 1, 2))(psnet_Int_feature)
    
    return Multiply()([input_feature, psnet_Int_feature])

def DDN_CustomFeatSelect(input_feature):
    kernel_size = 7
    
    if K.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        psnet_Int_feature = Permute((2,3,1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        psnet_Int_feature = input_feature
    
    avg_pool = Lambda(lambda DDN_init_param1: K.mean(x, axis=3, keepdims=True))(psnet_Int_feature)
    assert avg_pool.shape[-1] == 1
    max_pool = Lambda(lambda DDN_init_param1: K.max(x, axis=3, keepdims=True))(psnet_Int_feature)
    assert max_pool.shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat.shape[-1] == 2
    psnet_Int_feature = Conv2D(filters = 1,
                    kernel_size=kernel_size,
                    strides=1,
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer='he_normal',
                    use_bias=False)(concat)	
    assert psnet_Int_feature.shape[-1] == 1
    
    if K.image_data_format() == "channels_first":
        psnet_Int_feature = Permute((3, 1, 2))(psnet_Int_feature)
        
    return Multiply()([input_feature, psnet_Int_feature])

def DDN_PSNET(y, nb_channels, _strides=(1, 1), _project_shortcut=False):
    shortcut =DDN_init_param2

    
    DDN_init_param2 = Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
    DDN_init_param2 = BatchNormalization()(y)
    DDN_init_param2 = LeakyReLU()(y)

    DDN_init_param2 = Conv2D(nb_channels, kernel_size=(3, 3), strides=(1, 1), padding='same')(y)
    DDN_init_param2 = BatchNormalization()(y)

    if _project_shortcut or _strides != (1, 1):
        shortcut = Conv2D(nb_channels, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    DDN_init_param2 = Add()([shortcut,DDN_init_param2])
    DDN_init_param2 = LeakyReLU()(y)

    return DDN_init_param2

def DDN_Model_Gen():
    
    dropRate = 0.3
    
    init = Input(DDN_SHAPE)
    DDN_init_param1 = Conv2D(32, (3, 3), activation=None, padding='same')(init) 
    DDN_init_param1 = BatchNormalization()(x)
    DDN_init_param1 = Activation('relu')(x)
    DDN_init_param1 = Conv2D(32, (3, 3), activation=None, padding='same')(x) 
    DDN_init_param1 = BatchNormalization()(x)
    DDN_init_param1 = Activation('relu')(x)
    x1 = MaxPooling2D((2,2))(x)
    
    DDN_init_param1 = Conv2D(64, (3, 3), activation=None, padding='same')(x1)
    DDN_init_param1 = BatchNormalization()(x)
    DDN_init_param1 = Activation('relu')(x)
    DDN_init_param1 = DDN_PSNet_Block(x)
    DDN_init_param1 = DDN_PSNET(x, 64)
    x2 = MaxPooling2D((2,2))(x)
    
    DDN_init_param1 = Conv2D(128, (3, 3), activation=None, padding='same')(x2)
    DDN_init_param1 = BatchNormalization()(x)
    DDN_init_param1 = Activation('relu')(x)
    DDN_init_param1 = DDN_PSNet_Block(x)
    DDN_init_param1 = DDN_PSNET(x, 128)
    x3 = MaxPooling2D((2,2))(x)
    
    ginp1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(x1)
    ginp2 = UpSampling2D(size=(4, 4), interpolation='bilinear')(x2)
    ginp3 = UpSampling2D(size=(8, 8), interpolation='bilinear')(x3)
    
    hypercolumn = Concatenate()([ginp1, ginp2, ginp3]) 
    gap = GlobalAveragePooling2D()(hypercolumn)

    DDN_init_param1 = Dense(256, activation=None)(gap)
    DDN_init_param1 = BatchNormalization()(x)
    DDN_init_param1 = Activation('relu')(x)
    DDN_init_param1 = Dropout(dropRate)(x)
    
    DDN_init_param1 = Dense(256, activation=None)(x)
    DDN_init_param1 = BatchNormalization()(x)
    DDN_init_param1 = Activation('relu')(x)

    DDN_init_param2 = Dense(2, activation='softmax')(x)
   
    model = Model(init,DDN_init_param2)
    return model

model = DDN_Model_Gen()
model.summary()


def DDN_GenTest_DAT():
    gen = DDN_DS_PROCESS(DDN_SHAPE, DDN_BATCHsize, range(1), BASE_DIR, DDN_DS_Variant, DDN_NumSeed, DDN_TTrat, augment=False).Dataset_split_TT("test")
                       
    DDN_init_param1 = np.empty((len(gen[0]),)+DDN_SHAPE, dtype=np.float32)
    DDN_init_param2 = np.empty((len(gen[1]), 2), dtype=np.float32)
    
    for DDN_iter_param, path in tqdm(enumerate(gen[0])):
        DS_IMg = np.array(Image.open(gen[0][DDN_iter_param]))
        DS_IMg = resize(DS_IMg, DDN_SHAPE)

        DS_label = gen[1][DDN_iter_param]

        DDN_init_param1[DDN_iter_param] = DS_IMg
        DDN_init_param2[DDN_iter_param] = DS_label
        
    returnDDN_init_param1,DDN_init_param2

x,DDN_init_param2 = DDN_GenTest_DAT()
def threshold_arr(array):
    new_arr = []
    for DDN_iter_param, val in enumerate(array):
        loc = np.array(val).argmax(axis=0)
        k = list(np.zeros((len(val)), dtype=np.float32))
        k[loc]=1
        new_arr.append(k)
        
    return np.array(new_arr, dtype=np.float32)
models = []
for i in range(3):
    model = load_model("E:/PHD/shwetha_hyst_code/V6/trained_weights/200X_Mag.h5".format(i), custom_objects={'f1': f1, 'precision': precision, 'recall': recall})
    print(model.evaluate(x,DDN_init_param2, verbose=0))
    models.append(model)
def plot_confusion_matrix(cm,
                          target_names,
                          title='DDN Confusion matrix',
                          cmap=None,
                          normalize=True):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy


    if cmap is None:
        cmap = plt.get_cmap('Greys')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True DS_label')
    plt.xlabel('Predicted DS_label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig("200X - DDN confusion matrix.jpg", dpi=150)
    plt.show()

y_preds = threshold_arr(models[2].predict(x, verbose=0))

results = precision_recall_fscore_support(y, y_preds ,average='macro')
acc = accuracy_score(y, y_preds)

print("Accuracy: {}, F1_Score: {}, Precision: {}, Recall: {}".format(acc, results[2], results[0], results[1]))
print("\n")
print(classification_report(y, y_preds))
print("\n")
DDN_CMat = confusion_matrix(y.argmax(axis=1), y_preds.argmax(axis=1))

plot_confusion_matrix(cm           = DDN_CMat, 
                      normalize    = False,
                      target_names = ['BENIGN', 'MALIGNANT'],
                      title        = "DDN Confusion Matrix")

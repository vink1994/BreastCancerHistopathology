import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from albumentations import *
from skimage.transform import resize
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score
from tensorflow.keras.models import Model,load_model
from DS_UTILS import DDN_DS_PROCESS
from DDN_Ben_Model import *

DDN_SHAPE = (224, 224, 3)
DDN_BATCHsize = 24
Model_epoch = 100
DDN_InitSplits = 5
DDN_NumSeed = 9
DDN_TTrat = 0.4

proj_dir    = "E:/PHD/shwetha_hyst_code/mkfold/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/"
# FOR 40X
DDN_DS_Variant = "200X"
MDL_PATH = "E:/PHD/shwetha_hyst_code/V6/trained_weights/200X_Mag.h5"

DDN_GEN_REsCSV = "MAG" + DDN_DS_Variant +".csv"
DDN_GEN_REsFIG = "MAG" + DDN_DS_Variant + "DDN.jpg"

def DDN_GEN_InitData():
    gen = DDN_DS_PROCESS(DDN_SHAPE, DDN_BATCHsize, range(1), proj_dir , DDN_DS_Variant, DDN_NumSeed, DDN_TTrat, augment=False).get_pl_data("test")                
    DDN_init_param1 = np.empty((len(gen[0]),)+DDN_SHAPE, dtype=np.float32)
    DDN_init_param2 = np.empty((len(gen[1]), 2), dtype=np.float32)
    DDN_init_param3 = []
    I_name = []
    DDN_DG = []
    DS_ID =[]
    INit_Subcls=[]
    INit_CLS=[]
    
    for DDN_iter_param, path in tqdm(enumerate(gen[0])):
        DDN_IMG_PP= gen[0][DDN_iter_param]
        DDN_IMG_PP2 = DDN_IMG_PP.split("/")
        I_name.append(DDN_IMG_PP2[len(DDN_IMG_PP2)-1])
        DDN_DG.append(DDN_IMG_PP2[len(DDN_IMG_PP2)-2])
        DS_ID.append(DDN_IMG_PP2[len(DDN_IMG_PP2)-3])
        INit_Subcls.append(DDN_IMG_PP2[len(DDN_IMG_PP2)-4])
        INit_CLS.append(DDN_IMG_PP2[len(DDN_IMG_PP2)-6])
        DS_IMg = np.array(Image.open(DDN_IMG_PP))
        DS_IMg = resize(DS_IMg, DDN_SHAPE)
        DS_label = gen[1][DDN_iter_param]
        DDN_init_param3.append(DDN_IMG_PP)
        DDN_init_param1[DDN_iter_param] = DS_IMg
        DDN_init_param2[DDN_iter_param] = DS_label
        
    return DDN_init_param1,DDN_init_param2, DDN_init_param3, I_name, DDN_DG, DS_ID, INit_Subcls, INit_CLS


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

def threshold_arr(array):
    new_arr = []
    for DDN_iter_param, val in enumerate(array):
        loc = np.array(val).argmax(axis=0)
        k = list(np.zeros((len(val)), dtype=np.float32))
        k[loc]=1
        new_arr.append(k)
        
    return np.array(new_arr, dtype=np.float32)

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
    
    plt.savefig(DDN_GEN_REsFIG, dpi=150)
    plt.show()


x,DDN_init_param2, DDN_init_param3, I_name, DDN_DG, DS_ID, INit_Subcls, INit_CLS = DDN_GEN_InitData()

DATAFRAME_Gen_Param = pd.DataFrame(
    {'Class':INit_CLS,
     'Sub-Class':INit_Subcls,
     'Case_ID':DS_ID,
     'Magnification':DDN_DG,
     'Img_Name': I_name,
        'Img_Path':DDN_init_param3
    }
    )

DS_DF_PROCESSED =DATAFRAME_Gen_Param.Case_ID.unique()

DDN_model_st = DDN_Ben_Model()
DDN_model_st.summary()

DDN_model_st = load_model(MDL_PATH, custom_objects={'f1': f1, 'precision': precision, 'recall': recall})


pred_probab = DDN_model_st.predict(x, verbose=0)
y_preds = threshold_arr(pred_probab)

Res_Frme_Out = pd.DataFrame(
    {'Class':INit_CLS,
     'Sub-Class':INit_Subcls,
     'Case_ID':DS_ID,
     'Magnification':DDN_DG,
        'Img_Name': I_name,
     'GT':DDN_init_param2.argmax(axis=1),
     'PV': y_preds.argmax(axis=1)
    })
Res_Frme_Out.to_csv(DDN_GEN_REsCSV,index=False)
results = precision_recall_fscore_support(DDN_init_param2, y_preds ,average='macro')
acc = accuracy_score(DDN_init_param2, y_preds)

print("Metrics_Accuracy: {}, Metrics_F1_Score: {}, Metrics_Precision: {}, Metrics_Recall: {}".format(acc, results[2], results[0], results[1]))
print("\n")
print(classification_report(DDN_init_param2, y_preds))
print("\n")
DDN_CMat = confusion_matrix(DDN_init_param2.argmax(axis=1), y_preds.argmax(axis=1))




plot_confusion_matrix(cm           = DDN_CMat, 
                      normalize    = False,
                      target_names = ['BENIGN_class', 'MALIGNANT_class'],
                      title        = "DDN_Confusion_Matrix")
import tensorflow.keras.backend as K 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense,    \
                                    Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D, Input, Lambda, \
                                    LeakyReLU, MaxPooling2D, Multiply, Permute, Reshape, UpSampling2D   \

def DDN_customFeat_param1(pscstm_ft_vctr, ratio=8):

    pscstm_ft_vctr = DDN_customFeat_Input(pscstm_ft_vctr, ratio)
    pscstm_ft_vctr = DDN_customFeat_Input2(pscstm_ft_vctr)
    return pscstm_ft_vctr

def DDN_customFeat_Input(input_feature, ratio=8):
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
    
    pscstm_ft_vctr = Add()([avg_pool,max_pool])
    pscstm_ft_vctr = Activation('sigmoid')(pscstm_ft_vctr)

    if K.image_data_format() == "channels_first":
        pscstm_ft_vctr = Permute((3, 1, 2))(pscstm_ft_vctr)
    
    return Multiply()([input_feature, pscstm_ft_vctr])

def DDN_customFeat_Input2(input_feature):
    kernel_size = 7
    
    if K.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        pscstm_ft_vctr = Permute((2,3,1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        pscstm_ft_vctr = input_feature
    
    avg_pool = Lambda(lambda DDN_init_param1: K.mean(DDN_init_param1, axis=3, keepdims=True))(pscstm_ft_vctr)
    assert avg_pool.shape[-1] == 1
    max_pool = Lambda(lambda DDN_init_param1: K.max(DDN_init_param1, axis=3, keepdims=True))(pscstm_ft_vctr)
    assert max_pool.shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat.shape[-1] == 2
    pscstm_ft_vctr = Conv2D(filters = 1,
                    kernel_size=kernel_size,
                    strides=1,
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer='he_normal',
                    use_bias=False)(concat)	
    assert pscstm_ft_vctr.shape[-1] == 1
    
    if K.image_data_format() == "channels_first":
        pscstm_ft_vctr = Permute((3, 1, 2))(pscstm_ft_vctr)
        
    return Multiply()([input_feature, pscstm_ft_vctr])


def DDN_PSNET(DDN_init_param2, nb_channels, _strides=(1, 1), _project_shortcut=False):
    shortcut=DDN_init_param2
    

    DDN_init_param2 = Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(DDN_init_param2)
    DDN_init_param2 = BatchNormalization()(DDN_init_param2)
    DDN_init_param2 = LeakyReLU()(DDN_init_param2)

    DDN_init_param2 = Conv2D(nb_channels, kernel_size=(3, 3), strides=(1, 1), padding='same')(DDN_init_param2)
    DDN_init_param2 = BatchNormalization()(DDN_init_param2)

    if _project_shortcut or _strides != (1, 1):
        shortcut = Conv2D(nb_channels, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    DDN_init_param2 = Add()([shortcut,DDN_init_param2])
    DDN_init_param2 = LeakyReLU()(DDN_init_param2)

    return DDN_init_param2

def DDN_Ben_Model(input_shape=(224,224,3), n_classes=2):
 
        dropRate = 0.3
        init = Input(input_shape)
        DDN_init_param1 = Conv2D(32, (3, 3), activation=None, padding='same')(init) 
        DDN_init_param1 = BatchNormalization()(DDN_init_param1)
        DDN_init_param1 = Activation('relu')(DDN_init_param1)
        DDN_init_param1 = Conv2D(32, (3, 3), activation=None, padding='same')(DDN_init_param1) 
        DDN_init_param1 = BatchNormalization()(DDN_init_param1)
        DDN_init_param1 = Activation('relu')(DDN_init_param1)
        x1 = MaxPooling2D((2,2))(DDN_init_param1)
    
        DDN_init_param1 = Conv2D(64, (3, 3), activation=None, padding='same')(x1)
        DDN_init_param1 = BatchNormalization()(DDN_init_param1)
        DDN_init_param1 = Activation('relu')(DDN_init_param1)
        DDN_init_param1 = DDN_customFeat_param1(DDN_init_param1)
        DDN_init_param1 = DDN_PSNET(DDN_init_param1, 64)
        x2 = MaxPooling2D((2,2))(DDN_init_param1)
    
        DDN_init_param1 = Conv2D(128, (3, 3), activation=None, padding='same')(x2)
        DDN_init_param1 = BatchNormalization()(DDN_init_param1)
        DDN_init_param1 = Activation('relu')(DDN_init_param1)
        DDN_init_param1 = DDN_customFeat_param1(DDN_init_param1)
        DDN_init_param1 = DDN_PSNET(DDN_init_param1, 128)
        x3 = MaxPooling2D((2,2))(DDN_init_param1)
    
        ginp1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(x1)
        ginp2 = UpSampling2D(size=(4, 4), interpolation='bilinear')(x2)
        ginp3 = UpSampling2D(size=(8, 8), interpolation='bilinear')(x3)
    
        hypercolumn = Concatenate()([ginp1, ginp2, ginp3]) 
        gap = GlobalAveragePooling2D()(hypercolumn)

        DDN_init_param1 = Dense(256, activation=None)(gap)
        DDN_init_param1 = BatchNormalization()(DDN_init_param1)
        DDN_init_param1 = Activation('relu')(DDN_init_param1)
        DDN_init_param1 = Dropout(dropRate)(DDN_init_param1)
    
        DDN_init_param1 = Dense(256, activation=None)(DDN_init_param1)
        DDN_init_param1 = BatchNormalization()(DDN_init_param1)
        DDN_init_param1 = Activation('relu')(DDN_init_param1)

        DDN_init_param2 = Dense(n_classes, activation="softmax", name="PSMDL")(DDN_init_param1)
   
        model = Model(init,DDN_init_param2)
        return model
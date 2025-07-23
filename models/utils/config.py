config_2Dmf = {
    'name': 'mf',
    'token': 12,  # num tokens
    'embed': 64,  # embed dim
    'stem': 30,
    'bneck': {'e': 256, 'o': 128, 's': 1},  # exp out stride，e 扩展通道
    'body': [
        {'inp': 30, 'exp': 180, 'out': 64, 'se': None, 'stride': 1, 'heads': 2},
        {'inp': 64, 'exp': 256, 'out': 64, 'se': None, 'stride': 1, 'heads': 2},
        #{'inp': 48, 'exp': 96, 'out': 48, 'se': None, 'stride': 1, 'heads': 2},
        # stage4   
        #{'inp': 48, 'exp': 96, 'out': 64, 'se': None, 'stride': 2, 'heads': 2},
        #{'inp': 64, 'exp': 128, 'out': 64, 'se': None, 'stride': 1, 'heads': 2},
        # stage5   
        #{'inp': 64, 'exp': 128, 'out': 128, 'se': None, 'stride': 1, 'heads': 2},
        #{'inp': 96, 'exp': 192, 'out': 128, 'se': None, 'stride': 1, 'heads': 2},
    ],
    'fc1': 128  # hid_layer
    #'fc2': 1000  # num_clasess
    ,
}
config_3Dmf = {
    'name': '3Dmf',
    'token': 8,  # num tokens
    'embed': 24,  # embed dim
    'stem': 128,   
    'body': [
        {'inp': 1, 'exp': 8, 'out': 4, 'se': None, 'stride': 2, 'heads': 2},
        # stage3   
        {'inp': 4, 'exp': 16, 'out': 8, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 8, 'exp': 16, 'out': 8, 'se': None, 'stride': 1, 'heads': 2},
        # stage4   
        #{'inp': 8, 'exp': 24, 'out': 16, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 8, 'exp': 24, 'out': 16, 'se': None, 'stride': 1, 'heads': 2},
        # stage5   输出
        #{'inp': 32, 'exp': 64, 'out': 32, 'se': None, 'stride': 1, 'heads': 2},
        #{'inp': 96, 'exp': 192, 'out': 128, 'se': None, 'stride': 1, 'heads': 2},
    ],
    'fc1_in': 816,  # hid_layer 24*31+24+24
    'fc1': 128  # hid_layer
    #'fc2': 1000  # num_clasess
    ,
}
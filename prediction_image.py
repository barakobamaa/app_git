def pred_img(img_dir):
    # چک کردن اینکه ایا مدل قبلا دانلود شده یا نه
    # اگر نبود دانلود شود
    if(os.path.isfile('model/datset.csv')):
        print('model exists')
    
    else:
        print('model not found')
        url = 'https://drive.google.com/file/d/1PhSwy7YyUK3_pbwwp-6oUKcvJ6jAEbTU/view?usp=sharing'
        output = 'model/datset.csv'
        gdown.download(url, output, quiet=False) 
        print('model downloaded')
        
    image_size = 256
    
    from tensorflow.keras.models import load_model
    import pandas as pd
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    model = load_model('model/model_a.h5')
    

    #pic_file_name =  "243.jpg"
    #DATADIR = "/content/drive/My Drive/Kaggle/cepha400/cepha400/{}".format(pic_file_name)
    
    non_square_image = cv2.imread("input/{}".format(img_dir) ,cv2.IMREAD_GRAYSCALE)  # convert to array'
    
    
    #           -----    --- تغییرات برای خوراندن به مدل
    #   تغییرات کنتراست
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_contrast_img = clahe.apply(non_square_image)
    enhanced_contrast_img = non_square_image
    
    #--------    مربع کردن تصویر
    color = [0, 0, 0]
    print('tul= {}  arz={}'.format(enhanced_contrast_img.shape[0],enhanced_contrast_img.shape[1]))
    if(enhanced_contrast_img.shape[0]>enhanced_contrast_img.shape[1]):
      k=256/enhanced_contrast_img.shape[0]
      delta=enhanced_contrast_img.shape[0]-enhanced_contrast_img.shape[1]
      square = cv2.copyMakeBorder(enhanced_contrast_img, 0, 0, 0, delta, cv2.BORDER_CONSTANT,value=color)
      print('1')
    
    if(enhanced_contrast_img.shape[1]>enhanced_contrast_img.shape[0]):
      k=256/enhanced_contrast_img.shape[1]
      delta=enhanced_contrast_img.shape[1]-enhanced_contrast_img.shape[0]
      square = cv2.copyMakeBorder(enhanced_contrast_img, 0, delta,  0,0, cv2.BORDER_CONSTANT,value=color)
      print('2')
    
    if(enhanced_contrast_img.shape[1]==enhanced_contrast_img.shape[0]):
      k=256/enhanced_contrast_img.shape[1]
      delta=enhanced_contrast_img.shape[1]-enhanced_contrast_img.shape[0]
      square = cv2.copyMakeBorder(enhanced_contrast_img, 0, delta,  0,0, cv2.BORDER_CONSTANT,value=color)
      print('3')
    
    resized_square = cv2.resize(square, (image_size, image_size))  # resize to normalize data size
    reshaped_square = resized_square.reshape(-1,image_size,image_size, 1)
    # --------------------------------------------------------------------------
    #           پیش بینی
    first_prediction = model.predict([reshaped_square])
    
    
    #arz_correct = train_data_all.loc[train_data_all['image_path'] == pic_file_name][stp[0]]
    #tul_correct = train_data_all.loc[train_data_all['image_path'] == pic_file_name][stp[1]]
    #print(tul_correct)
    #print(arz_correct)
    
    
    first_arz=first_prediction[0][0]/k
    first_tul=first_prediction[0][1]/k
    print("predict: arz= {} tul= {}".format(first_arz,first_tul))
    #print("CORRECT: arz_correct= {} tul_correct= {}".format(arz_correct,tul_correct))
    
    plt.figure(figsize = (7,7))
    
    plt.scatter(first_arz,first_tul,color='r')
    
    plt.imshow(square, cmap='gray')  # graph it

    pred_dir='static/{}'.format(img_dir)
    plt.savefig(pred_dir)
     
    return img_dir
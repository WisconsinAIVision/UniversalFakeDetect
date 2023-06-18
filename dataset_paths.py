DATASET_PATHS = [



    # = = = = = = = = = = = = = = LDM = = = = = = = = = = = = = = = = #



    dict(
        real_path='../laion400m_data/val.pickle',
        fake_path='../FAKE_IMAGES/ldm_200step/val.pickle',
        data_mode='ours',
    ),

    dict(
        real_path='../laion400m_data/val.pickle',
        fake_path='../FAKE_IMAGES/ldm_200step_cfg3/val.pickle',
        data_mode='ours',
    ),




    # = = = = = = = = = = = = = = GLIDE = = = = = = = = = = = = = = = = #

    dict(
        real_path='../laion400m_data/val.pickle',
        fake_path='../FAKE_IMAGES/glide_100ddpm_27ddim/val.pickle',
        data_mode='ours',
    ),

    dict(
        real_path='../laion400m_data/val.pickle',
        fake_path='../FAKE_IMAGES/glide_75ddpm_27ddim/val.pickle',
        data_mode='ours',
    ),

    dict(
        real_path='../laion400m_data/val.pickle',
        fake_path='../FAKE_IMAGES/glide_50ddpm_27ddim/val.pickle',
        data_mode='ours',
    ),

    dict(
        real_path='../laion400m_data/val.pickle',
        fake_path='../FAKE_IMAGES/glide_50ddim_27ddim/val.pickle',
        data_mode='ours',
    ),

    dict(
        real_path='../laion400m_data/val.pickle',
        fake_path='../FAKE_IMAGES/glide_100ddpm_10ddim/val.pickle',
        data_mode='ours',
    ),


    # = = = = = = = = = = = = = = GUIDED = = = = = = = = = = = = = = = = #

    dict(
        real_path='../imagenet/val.pickle',
        fake_path='../FAKE_IMAGES/guided_imagenet_ddim25_cg1/val.pickle',
        data_mode='ours',
    ),



    # = = = = = = = = = = = = = = DALLE-MINI = = = = = = = = = = = = = = = = #

    dict(
        real_path='../laion400m_data/val.pickle',
        fake_path='../FAKE_IMAGES/DALLE-MINI',
        data_mode='ours',
    ),


    # = = = = = = = = = = = = = = CNN = = = = = = = = = = = = = = = = #


    dict(
        real_path='../FAKE_IMAGES/CNN/test/biggan/',   # Imagenet 
        fake_path='../FAKE_IMAGES/CNN/test/biggan/',
        data_mode='wang2020',
    ),

    dict(
        real_path='../FAKE_IMAGES/CNN/test/cyclegan',   
        fake_path='../FAKE_IMAGES/CNN/test/cyclegan',
        data_mode='wang2020',
    ),

    dict(
        real_path='../FAKE_IMAGES/CNN/test/gaugan',    # It is COCO 
        fake_path='../FAKE_IMAGES/CNN/test/gaugan',
        data_mode='wang2020',
    ),

    dict(
        real_path='../FAKE_IMAGES/CNN/test/progan',     
        fake_path='../FAKE_IMAGES/CNN/test/progan',
        data_mode='wang2020',
    ),


    dict(
        real_path='../FAKE_IMAGES/CNN/test/stylegan',    
        fake_path='../FAKE_IMAGES/CNN/test/stylegan',
        data_mode='wang2020',
    ),


    dict(
        real_path='../FAKE_IMAGES/CNN/test/whichfaceisreal',    # It is FFHQ 
        fake_path='../FAKE_IMAGES/CNN/test/whichfaceisreal',
        data_mode='wang2020',
    ),

    dict(
        real_path='../FAKE_IMAGES/CNN/test/crn',   # Images from some video games
        fake_path='../FAKE_IMAGES/CNN/test/crn',
        data_mode='wang2020',
    ),

    dict(
        real_path='../FAKE_IMAGES/CNN/test/imle',   # Images from some video games
        fake_path='../FAKE_IMAGES/CNN/test/imle',
        data_mode='wang2020',
    ),

    dict(
        real_path='../FAKE_IMAGES/CNN/test/stargan',  
        fake_path='../FAKE_IMAGES/CNN/test/stargan',
        data_mode='wang2020',
    ),

    dict(
        real_path='../FAKE_IMAGES/CNN/test/stylegan2',   
        fake_path='../FAKE_IMAGES/CNN/test/stylegan2',
        data_mode='wang2020',
    ),


    dict(
        real_path='../FAKE_IMAGES/CNN/test/deepfake',   
        fake_path='../FAKE_IMAGES/CNN/test/deepfake',
        data_mode='wang2020',
    ),


    dict(
        real_path='../FAKE_IMAGES/CNN/test/san',   
        fake_path='../FAKE_IMAGES/CNN/test/san',
        data_mode='wang2020',
    ),


    dict(
        real_path='../FAKE_IMAGES/CNN/test/seeingdark',   
        fake_path='../FAKE_IMAGES/CNN/test/seeingdark',
        data_mode='wang2020',
    ),



  
]

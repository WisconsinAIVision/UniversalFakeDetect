DATASET_PATHS = [



    # = = = = = = = = = = = = = = LDM = = = = = = = = = = = = = = = = #


    dict(
        real_path='../FAKE_IMAGES/CNN/test/progan',     
        fake_path='../FAKE_IMAGES/CNN/test/progan',
        data_mode='wang2020',
        key='progan'
    ),

    dict(
        real_path='../FAKE_IMAGES/CNN/test/cyclegan',   
        fake_path='../FAKE_IMAGES/CNN/test/cyclegan',
        data_mode='wang2020',
        key='cyclegan'
    ),

    dict(
        real_path='../FAKE_IMAGES/CNN/test/biggan/',   # Imagenet 
        fake_path='../FAKE_IMAGES/CNN/test/biggan/',
        data_mode='wang2020',
        key='biggan'
    ),


    dict(
        real_path='../FAKE_IMAGES/CNN/test/stylegan',    
        fake_path='../FAKE_IMAGES/CNN/test/stylegan',
        data_mode='wang2020',
        key='stylegan'
    ),


    dict(
        real_path='../FAKE_IMAGES/CNN/test/gaugan',    # It is COCO 
        fake_path='../FAKE_IMAGES/CNN/test/gaugan',
        data_mode='wang2020',
        key='gaugan'
    ),


    dict(
        real_path='../FAKE_IMAGES/CNN/test/stargan',  
        fake_path='../FAKE_IMAGES/CNN/test/stargan',
        data_mode='wang2020',
        key='stargan'
    ),


    dict(
        real_path='../FAKE_IMAGES/CNN/test/deepfake',   
        fake_path='../FAKE_IMAGES/CNN/test/deepfake',
        data_mode='wang2020',
        key='deepfake'
    ),


    dict(
        real_path='../FAKE_IMAGES/CNN/test/seeingdark',   
        fake_path='../FAKE_IMAGES/CNN/test/seeingdark',
        data_mode='wang2020',
        key='sitd'
    ),


    dict(
        real_path='../FAKE_IMAGES/CNN/test/san',   
        fake_path='../FAKE_IMAGES/CNN/test/san',
        data_mode='wang2020',
        key='san'
    ),


    dict(
        real_path='../FAKE_IMAGES/CNN/test/crn',   # Images from some video games
        fake_path='../FAKE_IMAGES/CNN/test/crn',
        data_mode='wang2020',
        key='crn'
    ),


    dict(
        real_path='../FAKE_IMAGES/CNN/test/imle',   # Images from some video games
        fake_path='../FAKE_IMAGES/CNN/test/imle',
        data_mode='wang2020',
        key='imle'
    ),
    

    dict(
        real_path='../imagenet/val.pickle',
        fake_path='../FAKE_IMAGES/guided_imagenet_ddim25_cg1/val.pickle',
        data_mode='ours',
        key='guided'
    ),


    dict(
        real_path='../laion400m_data/val.pickle',
        fake_path='../FAKE_IMAGES/ldm_200step/val.pickle',
        data_mode='ours',
        key='ldm_200'
    ),

    dict(
        real_path='../laion400m_data/val.pickle',
        fake_path='../FAKE_IMAGES/ldm_200step_cfg3/val.pickle',
        data_mode='ours',
        key='ldm_200_cfg'
    ),

    dict(
        real_path='../laion400m_data/val.pickle',
        fake_path='../FAKE_IMAGES/ldm_100step/val.pickle',
        data_mode='ours',
        key='ldm_100'
     ),


    dict(
        real_path='../laion400m_data/val.pickle',
        fake_path='../FAKE_IMAGES/glide_100ddpm_27ddim/val.pickle',
        data_mode='ours',
        key='glide_100_27'
    ),


    dict(
        real_path='../laion400m_data/val.pickle',
        fake_path='../FAKE_IMAGES/glide_50ddpm_27ddim/val.pickle',
        data_mode='ours',
        key='glide_50_27'
    ),


    dict(
        real_path='../laion400m_data/val.pickle',
        fake_path='../FAKE_IMAGES/glide_100ddpm_10ddim/val.pickle',
        data_mode='ours',
        key='glide_100_10'
    ),


    dict(
        real_path='../laion400m_data/val.pickle',
        fake_path='../FAKE_IMAGES/DALLE-MINI',
        data_mode='ours',
        key='dalle'
    ),



]

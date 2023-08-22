DATASET_PATHS = [


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
        real_path='./diffusion_datasets/imagenet',
        fake_path='./diffusion_datasets/guided',
        data_mode='wang2020',
        key='guided'
    ),


    dict(
        real_path='./diffusion_datasets/laion',
        fake_path='./diffusion_datasets/ldm_200',
        data_mode='wang2020',
        key='ldm_200'
    ),

    dict(
        real_path='./diffusion_datasets/laion',
        fake_path='./diffusion_datasets/ldm_200_cfg',
        data_mode='wang2020',
        key='ldm_200_cfg'
    ),

    dict(
        real_path='./diffusion_datasets/laion',
        fake_path='./diffusion_datasets/ldm_100',
        data_mode='wang2020',
        key='ldm_100'
     ),


    dict(
        real_path='./diffusion_datasets/laion',
        fake_path='./diffusion_datasets/glide_100_27',
        data_mode='wang2020',
        key='glide_100_27'
    ),


    dict(
        real_path='./diffusion_datasets/laion',
        fake_path='./diffusion_datasets/glide_50_27',
        data_mode='wang2020',
        key='glide_50_27'
    ),


    dict(
        real_path='./diffusion_datasets/laion',
        fake_path='./diffusion_datasets/glide_100_10',
        data_mode='wang2020',
        key='glide_100_10'
    ),


    dict(
        real_path='./diffusion_datasets/laion',
        fake_path='./diffusion_datasets/dalle',
        data_mode='wang2020',
        key='dalle'
    ),



]

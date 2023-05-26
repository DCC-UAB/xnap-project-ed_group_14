

def train_split(train_folder,test_folder,image_folder,text_file):
    image_names = []
    captions = []
    with open(text_file, 'r') as file:
        file.readline()
        for line in file:
            if '"' in line:
                list = line.strip().split('"')
                name = list[0][:-1]
                caption = []
                for i in list[1:]:
                    caption.extend(i)
                caption ='"' + ''.join(caption)+'"'
            else:
                name, caption = line.strip().split(',')
            image_names.append(name)
            captions.append(caption)
    num_images = len(image_names)
    num_train = int(num_images * 0.8)

    true=0
    for i in range(num_train):
        src_path = os.path.join(image_folder, image_names[i])
        dst_path = os.path.join(train_folder+'/Images', image_names[i])
        shutil.copy(src_path, dst_path)

        caption_file = os.path.join(train_folder, "captions.txt")
        with open(caption_file, "a") as file:
            if true == 0:
                file.write(f'image,caption\n')
                true = 1    
            file.write(f"{image_names[i]},{captions[i]}\n")
    true = 0
    for i in range(num_train, num_images):
        src_path = os.path.join(image_folder, image_names[i])
        dst_path = os.path.join(test_folder+'/Images', image_names[i])
        shutil.copy(src_path, dst_path)

        caption_file = os.path.join(test_folder, "captions.txt")
        
        with open(caption_file, "a") as file:
            if true == 0:
                file.write(f'image,caption\n')
                true = 1  
            file.write(f"{image_names[i]},{captions[i]}\n")




def crate_split():

    data_location = 'data'
    train_folder = data_location+"/train"
    test_folder = data_location+"/test"
    image_folder = data_location+"/Images"
    text_file = data_location+"/captions.txt"

    if not os.path.isdir(train_folder):
        if not os.path.exists(data_location+'/train'):
            os.makedirs(data_location+'/train')
            os.makedirs(data_location+'/train/Images')
        if not os.path.exists(data_location+'/test'):
            os.makedirs(data_location+'/test')
            os.makedirs(data_location+'/test/Images')
            
        train_split(train_folder,test_folder,image_folder,text_file)
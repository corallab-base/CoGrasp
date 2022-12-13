from PIL import Image

def merge_image():
    image1 = '/home/kaykay/isaacgym/python/contact_graspnet/RealData/IMG_0115.JPG'
    image3 = '/home/kaykay/isaacgym/python/contact_graspnet/RealData/IMG_0112.JPG'
    image2 = '/home/kaykay/isaacgym/python/contact_graspnet/RealData/1117_new.JPG'
    image4 = '/home/kaykay/isaacgym/python/contact_graspnet/RealData/1119_new.JPG'
    image1 = Image.open(image1).resize((400, 300))
    image2 = Image.open(image2).resize((400, 300))
    image3 = Image.open(image3).resize((400, 300))
    image4 = Image.open(image4).resize((400, 300))
    merge_a = Image.new(mode='RGB', size=(800, 300))
    merge_b = Image.new(mode='RGB', size=(800, 300))
    merge_a.paste(image1, (0, 0))
    merge_a.paste(image2, (400,0))
    merge_b.paste(image3, (0, 0))
    merge_b.paste(image4, (400, 0))
    merge_a.save('/home/kaykay/isaacgym/python/contact_graspnet/Paper/figure1a_new2.jpg')
    merge_b.save('/home/kaykay/isaacgym/python/contact_graspnet/Paper/figure1b_new2.jpg')

def merge_image_horizontal():
    cograsp_1 = '/home/kaykay/isaacgym/python/contact_graspnet/Paper/cograsp_1.png'
    cograsp_2 = '/home/kaykay/isaacgym/python/contact_graspnet/Paper/cograsp_2.png'
    cograsp_3 = '/home/kaykay/isaacgym/python/contact_graspnet/Paper/cograsp_3.png'
    cograsp_4 = '/home/kaykay/isaacgym/python/contact_graspnet/Paper/cograsp_4.png'
    cograsp_5 = '/home/kaykay/isaacgym/python/contact_graspnet/Paper/cograsp_5.png'
    cograsp_1 = Image.open(cograsp_1).resize((300, 200))
    cograsp_2 = Image.open(cograsp_2).resize((300, 200))
    cograsp_3 = Image.open(cograsp_3).resize((300, 200))
    cograsp_4 = Image.open(cograsp_4).resize((300, 200))
    cograsp_5 = Image.open(cograsp_5).resize((300, 200))
    merge_a = Image.new(mode='RGB', size=(1500, 200))
    merge_a.paste(cograsp_1, (0, 0))
    merge_a.paste(cograsp_2, (300, 0))
    merge_a.paste(cograsp_3, (600, 0))
    merge_a.paste(cograsp_4, (900, 0))
    merge_a.paste(cograsp_5, (1200, 0))
    merge_a.save('/home/kaykay/isaacgym/python/contact_graspnet/Paper/figure3a.jpg')

    graspnet_1 = '/home/kaykay/isaacgym/python/contact_graspnet/Paper/graspnet_1.png'
    graspnet_2 = '/home/kaykay/isaacgym/python/contact_graspnet/Paper/graspnet_2.png'
    graspnet_3 = '/home/kaykay/isaacgym/python/contact_graspnet/Paper/graspnet_3.png'
    graspnet_4 = '/home/kaykay/isaacgym/python/contact_graspnet/Paper/graspnet_4.png'
    graspnet_5 = '/home/kaykay/isaacgym/python/contact_graspnet/Paper/graspnet_5.png'
    graspnet_1 = Image.open(graspnet_1).resize((300, 200))
    graspnet_2 = Image.open(graspnet_2).resize((300, 200))
    graspnet_3 = Image.open(graspnet_3).resize((300, 200))
    graspnet_4 = Image.open(graspnet_4).resize((300, 200))
    graspnet_5 = Image.open(graspnet_5).resize((300, 200))
    merge_b = Image.new(mode='RGB', size=(1500, 200))
    merge_b.paste(graspnet_1, (0, 0))
    merge_b.paste(graspnet_2, (300, 0))
    merge_b.paste(graspnet_3, (600, 0))
    merge_b.paste(graspnet_4, (900, 0))
    merge_b.paste(graspnet_5, (1200, 0))
    merge_b.save('/home/kaykay/isaacgym/python/contact_graspnet/Paper/figure3b.jpg')

if __name__=='__main__':
    merge_image()

from quickdraw import QuickDrawData, QuickDrawDataGroup
from PIL import Image

qd = QuickDrawData()
apple = QuickDrawDataGroup("apple")
eye = QuickDrawDataGroup("eye")
octagon = QuickDrawDataGroup("octagon")
popsicle = QuickDrawDataGroup("popsicle")
pig = QuickDrawDataGroup("pig")

# input: rawImages - apple, eye, octagon, popsicle, or pig
#        numImages - number of training images you want for the input category
# output: a list of numImages images from rawImages category
#         as a 255*255 list of values (0 for white, 1 for black)
# example call: makeListOfDrawings(apple, 10)
def makeListOfDrawings(rawImages, numImages):
    listDrawings = []
    for i in range(numImages):
        draw = rawImages.get_drawing(i)
        img = draw.get_image(stroke_color=(0, 0, 0), stroke_width=2, bg_color=(255, 255, 255))
        pixels = list(img.getdata())
        output = []
        for j in range(len(pixels)):
            if (pixels[j] == (255, 255, 255)):
                output.append(0)
            elif (pixels[j] == (0, 0, 0)):
                output.append(1)
            else:
                output.append(-1)
        listDrawings.append(output)
    return listDrawings

print(len(makeListOfDrawings(apple, 10)))
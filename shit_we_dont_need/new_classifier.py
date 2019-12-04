from quickdraw import QuickDrawData, QuickDrawDataGroup
from PIL import Image
import numpy as np 

# input: rawImages - apple, eye, octagon, popsicle, or pig
#        numImages - number of training images you want for the input category
# output: a list of numImages images from rawImages category
#         as a 255*255 list of values (0 for white, 1 for black)
# example call: makeListOfDrawings(apple, 10)

def makeListOfDrawings(rawImages, numImages):
        list_drawings = []
        for i in range(numImages):
            draw = rawImages.get_drawing(i)
            drawing_name = draw.name
            stroke_list = draw.strokes 
            stroke_lengths = [len(stroke[0]) for stroke in stroke_list]
            total_points = sum(stroke_lengths)
            np_ink = np.zeros((total_points, 3), dtype=np.float32)
            current_t = 0 
            for stroke in stroke_list:
                for i in [0,1]:
                    np_ink[current_t:(current_t + len(stroke[0])), i] = stroke[i]
                current_t += len(stroke[0])
                np_ink[current_t-1, 2] = 1
            # Preprocessing.
            #1. Size normalization.
            lower = np.min(np_ink[:, 0:2], axis=0)
            upper = np.max(np_ink[:, 0:2], axis=0)
            scale = upper - lower
            scale[scale == 0] = 1
            np_ink[:, 0:2] = (np_ink[:, 0:2] - lower) / scale
            # 2. Compute deltas.
            np_ink[1:, 0:2] -= np_ink[0:-1, 0:2]
            np_ink = np_ink[1:, :]
            list_drawings.append((np_ink, drawing_name))
        print(list_drawings)
        return list_drawings

def all_drawings():
    ant = QuickDrawDataGroup("ant")
    bee = QuickDrawDataGroup("bee")
    eye = QuickDrawDataGroup("eye")
    hand = QuickDrawDataGroup("hand")
    pig = QuickDrawDataGroup("pig")
    drawings = [ant, bee, eye, hand, pig]
    all_drawings = []
    for drawing in drawings:
        all_drawings.append(makeListOfDrawings(drawing,500))
    return all_drawings

print(all_drawings())
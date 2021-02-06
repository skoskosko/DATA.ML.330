import csv
import cv2
from PIL import Image, ImageDraw



num = 2577

csv_file = []
with open('./Dry/labels.csv', 'r') as file:
    reader = csv.reader(file)
    first = True
    for row in reader:
        if first: first = False
        else: csv_file.append(row)


def draw_contours(filename):
    image = cv2.imread("./Dry/images/" + filename)
    color = (255, 0, 0) 
    for row in csv_file:
        if row[0] == filename:
            pt1 = ( int(row[4]), int(row[5]) )
            pt2 = ( int(row[6]), int(row[7]) )
            image = cv2.rectangle(image, pt1 , pt2, color, 5) 

    cv2.imwrite("./output/report/labels.%s" % filename, image)





draw_contours("dry_%s.png" % num)

imageColor = cv2.imread("./Dry/images/dry_%s.png" % num)
cv2.imwrite('./output/report/orig.dry_%s.png' % num, imageColor) 


from test_single import test as test_single_image

test_single_image(num)
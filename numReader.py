import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

import network

net = network.Network([784, 100, 30, 10])
# net.SGD(training_data, 20, 10, 3.0, test_data=test_data)
#net = network.Network([784, 10])
#net.SGD(training_data, 2, 10, 3.0, test_data=test_data)
net.loadVars()
net.saveVars()

import cv2
import numpy as np


#net = Network([784, 100, 30, 10])

# creating a 600 x 600 pixels canvas for mouse drawing
canvas = np.ones((600,600), dtype="uint8") * 255
canvas2 = np.zeros((280,280), dtype="uint8") * 255
# designating a 400 x 400 pixels point of interest on which digits will be drawn
canvas[100:500,100:500] = 0

start_point = None
end_point = None
is_drawing = False

def draw_line(img,start_at,end_at):
    cv2.line(img,start_at,end_at,255,25)

def on_mouse_events(event,x,y,flags,params):
    global start_point
    global end_point
    global canvas
    global is_drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        is_drawing = True
        start_point = (x,y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_drawing:
            end_point = (x,y)
            draw_line(canvas,start_point,end_point)
            start_point = end_point
    elif event == cv2.EVENT_LBUTTONUP:
        is_drawing = False


cv2.namedWindow("Test Canvas")
cv2.namedWindow("Image Canvas")
cv2.setMouseCallback("Test Canvas", on_mouse_events)


while(True):
    cv2.imshow("Test Canvas", canvas)
    cv2.imshow("Image Canvas", canvas2)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas[100:500,100:500] = 0
    elif key == ord('p'):
        image = canvas[100:500,100:500]
        input = cv2.resize(image, (20,20))
        input = np.pad(input, pad_width=4, mode='constant', constant_values=0).reshape((28,28,1)).astype('float32') / 255
        arr = np.reshape(np.array([input]), (784, 1))

        input = cv2.resize(input,(280,280))
        for r in range(0,len(input)):
            row = input[r]
            for c in range(0,len(row)):
                col = row[c]
                cv2.line(canvas2, (c,r), (c,r), int(input[r][c]*255), 1)
        #print(arr)
        result = net.guessNum(arr)

        print("PREDICTION : ",result)

cv2.destroyAllWindows()

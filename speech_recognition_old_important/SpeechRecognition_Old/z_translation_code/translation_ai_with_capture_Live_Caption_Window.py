from PIL import ImageGrab,Image
import numpy as np
import time
from io import BytesIO
import sys
from functools import partial
import webbrowser
import nltk as nltk
import srt
import que
import cv2
from cv2 import *
from moviepy.editor import *
import pytesseract
import traceback
import win32gui
import win32con
#import keyboard
#import speech_and_translat_facebook
from googletrans import Translator
import win32api
# import winGuiAuto

print("hello PyQt5")

from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QApplication,QLabel,QScrollArea,QWidget,QVBoxLayout,QLineEdit,QTextEdit,QFileDialog
from PyQt5.QtCore import Qt,QFile,QStringListModel

pytesseract.pytesseract.tesseract_cmd = r'D:\\Programs\\Tesseract-ocr_python\\install\\new\\tesseract.exe'


def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    # return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return Image.fromarray(img)

def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    # return cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)
    return np.asarray(img)

def replace_all(text, dic):
    for i, j in dic.iteritems():
        text = text.replace(i, j)
    return text

def filter_text_in_one_line(self, text_ocr):
    tex = text_ocr.replace("\n", " ")
    before_final = tex.replace("\n", " ")
    final = before_final.replace("|", "I")
    return final

def join_two_text_lines(string1, string2):
    string1 = string1
    string2 = string2
    # string1 = 'Hello how are you'
    # string2 = 'are you doing now? are you'
    i = 0
    while not string2.startswith(string1[i:]):
        i += 1
    sFinal = string1[:i] + string2
    # print(sFinal)

    return sFinal

def join_str(list_str):
    li = list_str

    if len(li) >= 2:
        f = ""
        for i, st in enumerate(li):
            # if i == len(li) - 1:
            #    break
            f = join_two_text_lines(f, st)

        print(f)
    return f

def extract_sentance(self, text):
    a_list = nltk.tokenize.sent_tokenize(text)
    full_text_from_sentances = ""
    for sen in a_list:
        full_text_from_sentances += sen + " "
    return a_list,full_text_from_sentances

def extract_word(self, text:str):
    all_words = text.replace(",","").replace(".","").split()
    return all_words
    # d = {",": "", ".": ""}
    # replace_all(text, d)

def return_image_to_pytesseract(fileName, ndarray: bool = False):
    if ndarray == False:
        image = cv2.imread(fileName)
    else:
        image = numpy.array(fileName)  # numpy.ndarray
    # plt.imshow(image)
    # cv2.imshow('image',image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]  # white background black font
    invert = 255 - thresh  # black background, white font
    # cv2.imshow('image', invert)
    im_black_background_white_font = Image.fromarray(np.uint8(cm.gist_earth(invert) * 255))
    im_white_background_black_font = Image.fromarray(np.uint8(cm.gist_earth(thresh) * 255))
    # im.show()
    return im_black_background_white_font, im_white_background_black_font

def opev(output):
    try:
        ii = Image.open(output)
        open_cv_image = np.array(ii)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()  # <class 'numpy.ndarray'>

        imbw, imwb = return_image_to_pytesseract(open_cv_image,
                                                 True)  # Image.fromarray(np.uint8(cm.gist_earth(invert) * 255))
        # imbw.show() #need this to pytesseract
    except Exception as e:
        print("read opencv from PIL.Image: ", e)
    text_ocr = pytesseract.image_to_string(imbw, lang='eng')  # lang='ara+eng')
    #final = filter_text_in_one_line(self, text_ocr)


class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    rec = False
    lis_images = []

    # def run(self):
    #     """Long-running task."""
    #     for i in range(5):
    #         sleep(1)
    #         self.progress.emit(i + 1)
    #     self.finished.emit()
    def captch_ex(self,file_name_opencv_ndarray):
        img = file_name_opencv_ndarray  # cv2.imread(file_name)

        img_final = file_name_opencv_ndarray  # cv2.imread(file_name)
        img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY)
        image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask)
        ret, new_img = cv2.threshold(image_final, 180, 255, cv2.THRESH_BINARY)  # for black text , cv.THRESH_BINARY_INV
        '''
                line  8 to 12  : Remove noisy portion 
        '''
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,
                                                             3))  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
        dilated = cv2.dilate(new_img, kernel, iterations=9)  # dilate , more the iteration more the dilation

        # for cv2.x.x

        contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)  # findContours returns 3 variables for getting contours
        print("con:", type(contours))
        # for cv3.x.x comment above line and uncomment line below

        # image, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

        area = None
        for contour in contours:
            # get rectangle bounding contour
            [x, y, w, h] = cv2.boundingRect(contour)

            # Don't plot small false positives that aren't text
            if w < 35 and h < 35:
                continue

            # draw rectangle around contour on original image
            # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            # im = Image.open(file_name)

            # print("type im: ",type(im))
            # v = Image.fromarray(im)
            area = (x, y, w + w, y + h)
            # cropped_img = im.crop(area)
            # cropped_img.show()

            '''
            #you can crop image and send to OCR  , false detected will return no text :)
            cropped = img_final[y :y +  h , x : x + w]

            s = file_name + '/crop_' + str(index) + '.jpg' 
            cv2.imwrite(s , cropped)
            index = index + 1

            '''
        # write original image with added contours to disk
        # cv2.imshow('captcha_result', img)
        # cv2.waitKey(0)
        return area
    def run(self):  # capture_specific_window(self):
        self.rec = True

        def mm(im1, im2):
            def fig2data(fig):

                # @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
                # @param fig a matplotlib figure
                # @return a numpy 3D array of RGBA values

                # draw the renderer
                fig.canvas.draw()

                # Get the RGBA buffer from the figure
                w, h = fig.canvas.get_width_height()
                buf = numpy.fromstring(fig.canvas.tostring_argb(), dtype=numpy.uint8)
                buf.shape = (w, h, 4)

                # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
                buf = numpy.roll(buf, 3, axis=2)
                return buf

            def fig_to_img(fig):
                # @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
                # @param fig a matplotlib figure
                # @return a Python Imaging Library ( PIL ) image
                # put the figure pixmap into a numpy array
                buf = fig2data(fig)
                w, h, d = buf.shape
                return Image.fromstring("RGBA", (w, h), buf.tostring())

            def fig2img(fig):
                # Convert a Matplotlib figure to a PIL Image and return it
                import io
                buf = io.BytesIO()
                fig.savefig(buf)
                buf.seek(0)
                img = Image.open(buf)
                return img

            print(PIL.__version__)

            def warpImages(img1, img2, H):

                rows1, cols1 = img1.shape[:2]
                rows2, cols2 = img2.shape[:2]

                list_of_points_1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
                temp_points = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)

                # When we have established a homography we need to warp perspective
                # Change field of view
                list_of_points_2 = cv2.perspectiveTransform(temp_points, H)

                list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

                [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
                [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

                translation_dist = [-x_min, -y_min]

                H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

                output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max - x_min, y_max - y_min))
                output_img[translation_dist[1]:rows1 + translation_dist[1],
                translation_dist[0]:cols1 + translation_dist[0]] = img1

                return output_img

            def draw_matches(img1, keypoints1, img2, keypoints2, matches):
                r, c = img1.shape[:2]
                r1, c1 = img2.shape[:2]

                # Create a blank image with the size of the first image + second image
                output_img = np.zeros((max([r, r1]), c + c1, 3), dtype='uint8')
                output_img[:r, :c, :] = np.dstack([img1, img1, img1])
                output_img[:r1, c:c + c1, :] = np.dstack([img2, img2, img2])

                # Go over all of the matching points and extract them
                for match in matches:
                    img1_idx = match.queryIdx
                    img2_idx = match.trainIdx
                    (x1, y1) = keypoints1[img1_idx].pt
                    (x2, y2) = keypoints2[img2_idx].pt

                    # Draw circles on the keypoints
                    cv2.circle(output_img, (int(x1), int(y1)), 4, (0, 255, 255), 1)
                    cv2.circle(output_img, (int(x2) + c, int(y2)), 4, (0, 255, 255), 1)

                    # Connect the same keypoints
                    cv2.line(output_img, (int(x1), int(y1)), (int(x2) + c, int(y2)), (0, 255, 255), 1)

                return output_img

            img1 = im1
            img2 = im2
            # eoo = cv2.imread("D:/self/stitch/2.jpg")
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            print("type gray: ", type(img2_gray))
            orb = cv2.ORB_create(nfeatures=2000)
            # Find the key points and descriptors with ORB
            keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
            keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

            # Create a BFMatcher object.
            # It will find all of the matching keypoints on two images
            bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)
            # Find matching points
            matches = bf.knnMatch(descriptors1, descriptors2, k=2)

            all_matches = []
            for m, n in matches:
                all_matches.append(m)

            img3 = draw_matches(img1_gray, keypoints1, img2_gray, keypoints2, all_matches[:30])
            # plt.imshow(img3)
            # plt.title('img_read_as_grayscale')
            # plt.show()
            print("type img3: ", type(img3))
            # cv2.imshow("ff", img3)
            # cv2.waitKey(0)

            # plt.figure()
            #
            # ...
            #
            # plt.imshow(image)
            #
            # # remove white padding
            # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            # plt.axis('off')
            # plt.axis('image')
            #
            # # redraw the canvas
            # fig = plt.gcf()
            # fig.canvas.draw()
            #
            # # convert canvas to image using numpy
            # img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            # img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            #
            # # opencv format
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            #
            # plt.close()

            # @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
            # @param fig a matplotlib figure
            # @return a numpy 3D array of RGBA values

            #     # draw the renderer
            #     fig.canvas.draw()
            #
            #     # Get the RGBA buffer from the figure
            #     w, h = fig.canvas.get_width_height()
            #     buf = numpy.fromstring(fig.canvas.tostring_argb(), dtype=numpy.uint8)
            #     buf.shape = (w, h, 4)
            #
            #     # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
            #     buf = numpy.roll(buf, 3, axis=2)
            #     return buf
            # def fig_to_img(fig):
            #     #@brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
            #     #@param fig a matplotlib figure
            #     #@return a Python Imaging Library ( PIL ) image
            #     # put the figure pixmap into a numpy array
            #     buf = fig2data(fig)
            #     w, h, d = buf.shape
            #     return Image.fromstring("RGBA", (w, h), buf.tostring())

            good = []
            for m, n in matches:
                if m.distance < 0.6 * n.distance:
                    good.append(m)
            MIN_MATCH_COUNT = 10
            if len(good) > MIN_MATCH_COUNT:
                # Convert keypoints to an argument for findHomography
                src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                # Establish a homography
                M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                # cv2.imshow("hh",M)
                # cv2.waitKey(0)
                result = warpImages(img2, img1, M)

                print("type result: ", type(result))
                # plt.imshow(result)
                # plt.title("ffgg")
                # plt.show()
                # plt.savefig("D:/self/stitch/new.jpg")

                img = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                # cv2.cv2.imwrite("D:/self/stitch/full.jpg",img)
                # cv2.imshow('img', img)
                # cv2.waitKey(0)
                return img

        try:

            #q = que.queue_se()
            i = 0

            while self.rec:
                # QThread.sleep(1)
                # self.repaint()
                # self.update()
                # self.setUpdatesEnabled(False)
                hwnd = win32gui.FindWindow(None, r"Live Caption")
                if win32gui.IsWindowVisible(hwnd):
                    print("visable")
                    QThread.msleep(100)
                    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                          win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
                    rect = win32gui.GetWindowPlacement(hwnd)[-1]
                    n1 = rect[0] + 20  # left
                    n2 = rect[1] + 75    #37  # top
                    n3 = rect[2] - 25  # right
                    n4 = rect[3] - 47
                    t = (n1, n2, n3, n4)
                    print(t)
                    # (510, 248, 1213, 582)
                    # rr = win32gui.GetWindowPlacement(hwnd)
                    # print("rect: " + str(rect))
                    # print("rr: " + str(rr))
                    image = ImageGrab.grab(t)

                    # image_nd = convert_from_image_to_cv2(image)
                    try:
                        pil_image = image.convert('RGB')
                        # pil_image.size[0]
                        open_cv_image = numpy.array(pil_image)
                        # Convert RGB to BGR
                        image_nd = open_cv_image[:, :, ::-1].copy()

                        area1 = self.captch_ex(image_nd)
                        # crop = image[y:y+h, x:x+w]
                        print("image_nd: ",type(image_nd))
                        crop = image_nd[area1[1]:area1[3], 0:pil_image.size[0]]
                    except Exception as e:
                        print("crop: ", e)
                    #cv2.imshow('Image', crop)
                    #cv2.waitKey(0)
                    # cv2.imshow("Image",image_nd)
                    # time.sleep(5)

                    #imbw, imwb = return_image_to_pytesseract(image_nd, True)
                    #text_ocr = pytesseract.image_to_string(imbw, lang='eng')

                    #print("text_ocr_count:",str(text_ocr).count("\n"))
                    #name = "D:/self/stitch/" + str(i) + ".jpg"
                    #if str(text_ocr).count("\n") > 7:
                    self.lis_images.append(crop)

                    # cv2.cv2.imshow("image",image_nd)
                    # #cv2.cv2.imwrite(name, image_nd)
                    # cv2.cv2.waitKey(1)
                    # if i == 20:
                    #     break
                    """
                    if q.isFull():
                        newim = mm(q.getFrontQueue(), q.getRearQueue())
                        q.setFrontQueue(newim)
                        q.deleteQueue()
                    else:
                        q.addQueue(image_nd)
                    """

                    # time.sleep(2)
                    # if keyboard.is_pressed('q'):
                    #     rec = False
                    # if i == 5:
                    #     cv2.cv2.imshow("imf", q.getFrontQueue())
                    #     cv2.cv2.waitKey(0)
                    #     exit()
                    """
                    if self.rec == False:
                        if not q.isEmpty():

                            cv2.cv2.imshow("imf", q.getFrontQueue())
                            cv2.cv2.waitKey(0)
                        else:
                            # exit()
                            break
                    """
                    i += 1
                    print(i)
                    QThread.msleep(500)
                    # self.updatesEnabled(True)
                    """
                    if not q.isFull():
                        q.addQueue(image_nd)
                        #q.addQueue("string_2 ")
                    if q.isFull():
                        image_Stitcher = cv2.Stitcher_create()
                        images = [q.getFrontQueue(),q.getRearQueue()]
                        error,stithed_image = image_Stitcher.stitch(images)
                        if not error:
                            print("type stithed_image: ", type(stithed_image))
                            q.setFrontQueue(stithed_image)
                            print("type getFrontQueue: ", type(q.getFrontQueue()))
                            # print(q.getFrontQueue())
                            # print("\n")
                            # print(q.getRearQueue())
                            q.deleteQueue()
                            cv2.imshow("image", q.getFrontQueue())
                            cv2.waitKey(0)
                        else:
                            print("error stitch: ",error," ", stithed_image)
                            exit()
                        """
                    # image.show()
                    # image_list.append(image)
                else:
                    print("not visable")
                    QThread.sleep(1)
                    # self.update()
                    # self.updatesEnabled()
                    # self.setUpdatesEnabled(True)
                    # self.setEnabled(False)
        except Exception as e:
            print(e)
        self.finished.emit()

class App(QtWidgets.QMainWindow, QThread):
    # change_text = QtCore.pyqtSignal()
    worker = None

    def __init__(self):
        super().__init__()

        filepath = r'C:/Users/User/Desktop/self/tr/STEP.txt'
        self.lis_images = []
        self.rec = True
        self.initUI()
        self.topMenu()

        self.textbox = QTextEdit(self)
        self.textbox.move(20, 100)
        self.textbox.resize(300, 100)
        self.textbox.selectionChanged.connect(self.handleSelectionChanged)
        # self.textbox.setAlignment(QtCore.Qt.AlignLeft) #AlignRight)
        #self.textbox.textChanged.connect(self.onChange)
        self.textbox.installEventFilter(self)
        #self.textbox.focusChanged.connect(self.onChange)
        self.typing_timer = QtCore.QTimer()
        self.typing_timer.setSingleShot(True)
        self.typing_timer.timeout.connect(self.make_changes)
        self.textbox.textChanged.connect(self.start_typing_timer)


        self.textbox_tr = QTextEdit(self)
        self.textbox_tr.move(330, 100)
        self.textbox_tr.resize(300, 100)
        self.textbox.setAlignment(QtCore.Qt.AlignLeft)  # AlignRight)
        # self.textbox_tr.textChanged.connect(self.onChange)



        #
        self.textbox_origin = QTextEdit(self)
        self.textbox_origin.resize(230, 30)
        self.textbox_origin.move(5, 25)
        #self.textbox_origin.selectionChanged.connect(self.handleSelectionChanged)
        # self.textbox.setAlignment(QtCore.Qt.AlignLeft) #AlignRight)
        # self.textbox.textChanged.connect(self.onChange)
        self.textbox_origin.installEventFilter(self)
        # self.textbox.focusChanged.connect(self.onChange)
        self.typing_timer_origin = QtCore.QTimer()
        self.typing_timer_origin.setSingleShot(True)
        self.typing_timer_origin.timeout.connect(self.make_changes_dest)
        self.textbox_origin.textChanged.connect(self.start_typing_timer_origin)
        #
        # self.textbox_origin = QTextEdit(self)
        # self.textbox_origin.resize(230, 30)
        # self.textbox_origin.move(5, 25)

        self.text = None
        self.textbox_dest = QTextEdit(self)
        self.textbox_dest.resize(230, 30)
        self.textbox_dest.move(5, 55)
        self.textbox_dest.setAlignment(QtCore.Qt.AlignLeft)
        # self.textbox_tr.resize(300, 30)
        #self.textbox.setAlignment(QtCore.Qt.AlignLeft)  # AlignRight)

        # self.label_origin = QLabel('origin of google any  dsad dafav fdsv ', self)
        # self.label_origin.move(20, 20)
        # self.textbox_tr.resize(300, 100)
        # self.dest = QLabel('السلام عليكم ورحمة الله وبركاته ', self)
        # self.dest.move(100, 40)

        #self.dest.setAlignment(QtCore.Qt.AlignLeft)
        # self.button = QtWidgets.QPushButton('open browser translator')
        # self.button.move(20, 30)
        # self.button.resize(150, 50)
        # self.layout().addWidget(self.button)
        # self.button.clicked.connect(self.open_browser)

        self.select_image_English = QtWidgets.QPushButton('select image English')
        self.select_image_English.move(250, 30)
        self.select_image_English.resize(130, 50)
        self.layout().addWidget(self.select_image_English)
        self.select_image_English.clicked.connect(self.choose_file_image_english)

        self.current_translate = QtWidgets.QPushButton('current capture translate')
        self.current_translate.move(400, 30)
        self.current_translate.resize(130, 50)
        self.layout().addWidget(self.current_translate)
        self.current_translate.clicked.connect(self.capture_specific_window_current)

        self.send_text_to_file_as_words = QtWidgets.QPushButton('send text to file as words')
        self.send_text_to_file_as_words.move(20, 200)
        self.send_text_to_file_as_words.resize(170, 50)
        self.layout().addWidget(self.send_text_to_file_as_words)
        self.send_text_to_file_as_words.clicked.connect(partial(self.add_words_to_file, filepath))

        self.filter_STEP_txt = QtWidgets.QPushButton('filter STEP.txt')
        self.filter_STEP_txt.move(200, 200)
        self.filter_STEP_txt.resize(170, 50)
        self.layout().addWidget(self.filter_STEP_txt)
        self.filter_STEP_txt.clicked.connect(partial(self.filter_file_txt_duplicate, filepath))

        self.add_sentence_to_file_STEP_txt = QtWidgets.QPushButton('add sentence to file STEP.txt')
        self.add_sentence_to_file_STEP_txt.move(200, 250)
        self.add_sentence_to_file_STEP_txt.resize(200, 50)
        self.layout().addWidget(self.add_sentence_to_file_STEP_txt)
        self.add_sentence_to_file_STEP_txt.clicked.connect(partial(self.add_sentence_to_file, filepath))
        try:
            self.rec_panorama = QtWidgets.QPushButton('rec panorama')
            self.rec_panorama.move(10, 250)
            self.rec_panorama.resize(100, 50)
            self.layout().addWidget(self.rec_panorama)
            self.rec_panorama.clicked.connect(self.runLongTask)

            self.stop_panorama = QtWidgets.QPushButton('stop panorama')
            self.stop_panorama.move(100, 250)
            self.stop_panorama.resize(100, 50)
            self.layout().addWidget(self.stop_panorama)
            self.stop_panorama.clicked.connect(self.stop_panoramic)


        except Exception as e:
            print("Ex __init__:  ", e)

        # time.sleep(3)
        # self.button.click()
    def translate_google(self,text):
        translator = Translator()
        translation = translator.translate(text, dest='ar')
        #print(translation.text)
        return translation.text
    def make_changes_dest(self):
        print("run translate")
        print("origin ", self.textbox_origin.toPlainText())
        if self.textbox_origin.toPlainText() != "" and self.textbox_origin.toPlainText() != None:
            start_time = time.time()

            en_ar = self.translate_google(self.textbox_origin.toPlainText())  # speech_and_translat_facebook.translate_en_to_ar(self.textbox.toPlainText())
            print("--- %s seconds ---" % (time.time() - start_time))

            # self.textbox_dest.setText(en_ar)
            self.textbox_dest.setText(en_ar)

        else:
            print("none or null")
    def handleSelectionChanged(self):
        self.text = self.textbox.textCursor()

        if self.text.hasSelection():
            if(self.text.selectedText() != "" and self.text.selectedText()  != None):
                self.text = self.textbox.textCursor().selectedText()

            # process text here...
            # self.setToolTip(str(text))

            self.textbox_origin.setText(self.text)

        #print("selected text:: ", self.text)
        #self.textbox_origin.setText(text)

        #self.textbox_dest.setText(self.translate_google(text))
    def start_typing_timer_origin(self):
        """Wait until there are no changes for 1 second before making changes."""
        self.typing_timer_origin.start(1000)
    def start_typing_timer(self):
        """Wait until there are no changes for 1 second before making changes."""
        self.typing_timer.start(1000)

    def make_changes(self):
        #txt = self.txt_edit.toPlainText()
        print("run translate")
        if self.textbox.toPlainText() != "" and self.textbox.toPlainText() != None:
            start_time = time.time()
            en_ar = self.translate_google(self.textbox.toPlainText()) #speech_and_translat_facebook.translate_en_to_ar(self.textbox.toPlainText())
            print("--- %s seconds ---" % (time.time() - start_time))

            #self.textbox_dest.setText(en_ar)
            self.textbox_tr.setText(en_ar)
            #self.textbox_dest.setText(en_ar)

        # RUN SQL OR DO SOMETHING ELSE WITH THE TEXT HERE #
    def capture_specific_window_current(self):
        import numpy as np
        import cv2
        from PIL import ImageGrab,Image
        import win32api
        # import winGuiAuto
        import win32gui
        import win32con
        hwnd = win32gui.FindWindow(None, r"Live Caption")
        if win32gui.IsWindowVisible(hwnd):
            print("visable")
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                  win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
            rect = win32gui.GetWindowPlacement(hwnd)[-1]
            # (510, 248, 1213, 582)
            n1 = rect[0] + 20  # left
            n2 = rect[1] + 37  # top
            n3 = rect[2] - 25  # right
            n4 = rect[3] - 47
            t = (n1, n2, n3, n4)
            #print(t)
            #rr = win32gui.GetWindowPlacement(hwnd)
            #print("rect: " + str(rect))
            #print("rr: " + str(rr))
            image = ImageGrab.grab(t)
            #image.show()
            output = BytesIO()
            image.save(output, format="JPEG")
            #return output
            self.current_trans(output)
    def current_trans(self,fileName):
        try:
            image = convert_from_image_to_cv2(Image.open(fileName))
            imbw,imwb = return_image_to_pytesseract(image,True)
            #cv2.imshow("imbw",image)
            #imbw.show()
            #image = cv2.imread(Image.open(fileName))

            #image = fileName  # numpy.ndarray
            #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #imbw, imwb = return_image_to_pytesseract(im,True)  # Image.fromarray(np.uint8(cm.gist_earth(invert) * 255))
            # imbw.show() need this to pytesseract
            # imwb.show()
            text_ocr = pytesseract.image_to_string(imbw, lang='eng')  # lang='ara+eng')
        except Exception as e:
            print("current_trans: ",e)
        final = filter_text_in_one_line(self, str(text_ocr).replace("Live Caption ","").replace("Live Caption",""))
        #a_list, full_text_from_sentances = extract_sentance(self, final)

        #en_ar = speech_and_translat_facebook.translate_en_to_ar(final)
        #self.textbox_tr.setText(en_ar)

        self.textbox.setText(final)
    def eventFilter(self, source, event):
        if (event.type() == QtCore.QEvent.FocusOut and
                source is self.textbox):
            print('eventFilter: focus out')
            # return true here to bypass default behaviour
        return super(App, self).eventFilter(source, event)

    def my_handler(self, event):
        QtWidgets.QTextEdit.focusOutEvent(self.textbox, event)
        print("focus out")
        # my own handling follows...
    def onChange(self):
        # print(self.textbox.toPlainText())
        print("OnChange out")
        print("focus: ", self.textbox.hasFocus())
        if self.textbox.hasFocus() == False:
            if self.textbox.toPlainText() != "" and self.textbox.toPlainText() != None:
                print("OnChange inside")
                en_to_ar = self.translate_google(self.textbox.toPlainText()) #speech_and_translat_facebook.translate_en_to_ar(self.textbox.toPlainText())
                #self.textbox.textCursor().selectedText()
                self.textbox_tr.setText(str(en_to_ar))
        print("focus finish: ", self.textbox.hasFocus())
        # if self.textbox.toPlainText() != "" and self.textbox.toPlainText() != None:
        #     print("OnChange inside")
        #     en_to_ar = speech_and_translat_facebook.translate_en_to_ar(self.textbox.toPlainText())
        #     self.textbox_tr.setText(str(en_to_ar))
        # else:
        #     print("empty or None")

    def runLongTask(self):
        try:
            # Step 2: Create a QThread object
            self.thread = QThread()
            # Step 3: Create a worker object
            self.worker = Worker()
            # Step 4: Move worker to the thread
            self.worker.moveToThread(self.thread)
            # Step 5: Connect signals and slots
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            self.worker.progress.connect(self.reportProgress)
            # Step 6: Start the thread
            self.thread.start()

            # Final resets
            self.rec_panorama.setEnabled(False)
            self.thread.finished.connect(
                lambda: self.rec_panorama.setEnabled(True)
            )
            self.thread.finished.connect(
                self.stop_panoramic
            )
        except Exception as e:
            import sys, os
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print("Ex runLongTask: ", e)

    def reportProgress(self):
        pass

    def stop_panoramic(self):
        self.worker.rec = False
        if len(self.worker.lis_images) > 2:
            #list_text = []
            for img in self.worker.lis_images:
                #imbw, imwb = return_image_to_pytesseract(img, True)  # Image.fromarray(np.uint8(cm.gist_earth(invert) * 255))
                #text_ocr = pytesseract.image_to_string(imbw, lang='eng')  # lang='ara+eng')
                #final = filter_text_in_one_line(self, text_ocr)
                #print("final regular:" , final)
                #list_text.append(final)

                # img = cv2.imread(filename)
                height, width, layers = img.shape
                size = (width, height)

            #final_finish = join_str(list_text)
            #print("final finish:", final_finish)
            #cv2.ROTATE_90_COUNTERCLOCKWISE
            # with open("live.txt","w") as f:
            #     f.write(final_finish)
            #     f.close()
            clips = [ImageClip(m).set_duration(2).set_pos("left","top") for m in self.worker.lis_images]

            concat_clip = concatenate_videoclips(clips, method="compose")
            concat_clip.write_videofile("D:/self/stitch/test.mp4", fps=5)
            out = cv2.VideoWriter('D:/self/stitch/project.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 1, size,True)

            for i in range(len(self.worker.lis_images)):
                out.write(self.worker.lis_images[i])
                cv2.imwrite("D:/self/stitch/image/" +str(i)+".jpg" , self.worker.lis_images[i])
            out.release()
            print("len self.worker.lis_images: ", len(self.worker.lis_images))
        else:
            print("not image in list")
        self.worker.lis_images = []

    def filter_file_txt_duplicate(self, filename=r'D:/self/Pycharm/Projects/transvideo/STEP.txt'):
        li = []

        try:
            with open(filename, "r", encoding='utf-8') as file:
                for line in file:
                    li.append(line)
            print("first count lines : " + str(len(li)))
            final_test = list(set(li))
            print("set list count: " + str(len(final_test)))
        except Exception as e:
            print("filter_file_txt_duplicate", e)

        # print(final_test)
        with open(filename, "w", encoding='utf-8') as file:
            for w in final_test:
                file.write(w)

    def choose_file_image_english(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(None, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*);;Python Files (*.py);;Image jpg (*.jpg);;Image jpeg (*.jpeg);;Image png (*.png)",
                                                  options=options)
        print(fileName)

        try:

            # # im = Image.open(fileName) #correct
            #
            # image = cv2.imread(fileName)
            # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] # background white , font black
            # invert = 255 - thresh #black background, white font
            #
            # #cv2.imshow('image', invert)
            #
            # im = Image.fromarray(np.uint8(cm.gist_earth(invert)*255))
            # im.show()

            imbw, imwb = return_image_to_pytesseract(fileName)  # Image.fromarray(np.uint8(cm.gist_earth(invert) * 255))
            # imbw.show() need this to pytesseract
            # imwb.show()
            text_ocr = pytesseract.image_to_string(imbw, lang='eng')  # lang='ara+eng')

            final = filter_text_in_one_line(self, text_ocr)
            a_list, full_text_from_sentances = extract_sentance(self, final)
            print("final: " + "\n" + final)
            # print("\n" + "a_list: " + "\n" , a_list)
            # print("\n" + "full_text_from_sentances: " + "\n" + full_text_from_sentances)
            # print("\n" + "words: " + "\n" , extract_word(self,final))
            self.textbox.setText(final)
            # words = extract_word(self, final)
            # file = open("STEP.txt", "a")
            # te = ""
            # for wor in words:
            #     te += wor + "\n"
            #
            # file.writelines(te)

            # s = "didn't"

            # st =  "hello, my friend."
            # print(st.replace(",","").replace(".","").split())

            # t = s.replace(".","")
            # print(t)
            # print(final)
        except Exception as e:
            print("error pytessaarct", e)

    def add_sentence_to_file(self, filepath):
        file = open(filepath, "a", encoding='utf-8')
        try:

            te = self.textbox.toPlainText()
            # tokens = nltk.word_tokenize(te)
            final = filter_text_in_one_line(self, te)
            sentances, full_string = extract_sentance(self, final)

            print("sentances from textbox: ", sentances)
            # �’

            # file = open("STEP.txt", "a", encoding='utf-8')
            te = ""
            for wor in sentances:
                te += wor + "\n"
            file.writelines(te)
            file.close()
        except Exception as e:
            print(e)
            #import traceback
            print(traceback.format_exc())

    def add_words_to_file(self, filepath):
        # url = "https://translate.google.com/?pli=1&sl=en&tl=ar&text=" + self.textbox.toPlainText()
        # webbrowser.open(url)
        # file = QFile("STEP.txt", "a")

        file = open(filepath, "a")

        origin = self.textbox_origin.toPlainText()
        dest = self.textbox_dest.toPlainText()

        file = open("STEP.txt", "a", encoding='utf-8')
        te = origin + " -> " + dest + "\n"
        file.writelines(te)
        file.close()
        """
        try:
            te = self.textbox.toPlainText()
            # tokens = nltk.word_tokenize(te)
            final = filter_text_in_one_line(self, te)
            words = extract_word(self, final)
        except Exception as e:
            print("Ex add_words_to_file: ", e)
        print("words from textbox: ", words)
        # �’

        file = open("STEP.txt", "a", encoding='utf-8')
        te = ""
        for wor in words:
            te += wor + "\n"
        file.writelines(te)
        file.close()
        """

    def press(self):
        self.button.click()

    def set_text(self, text):
        self.textbox.setText(text)

    def pr(self):
        print("show im")
        return self.textbox

    def open_browser(self):
        url = "https://translate.google.com/?pli=1&sl=en&tl=ar&text=" + self.textbox.toPlainText()
        webbrowser.open_new(url)
        # self.textbox.setText("button")

        """
        # v = QLabel()
        # v.destroy()
        # v.close()
        #my self
        self.scroll = QScrollArea()  # Scroll Area which contains the widgets, set as the centralWidget
        self.widget = QWidget()  # Widget that contains the collection of Vertical Box
        self.vbox = QVBoxLayout()  # The Vertical Box that contains the Horizontal Boxes of  labels and buttons
        # creating a label widget
        # by default label will display at top left corner
        text = None
        with open('redirect.txt', 'r') as file:
            text = file.read()



        tex = text.replace("\n", " ")
        before_final = tex.replace("\n", " ")
        final = before_final.replace("|","I")
        a_list = nltk.tokenize.sent_tokenize(final)
        print(a_list)

        # for host in a_list:
        #     label = QLabel(host)
        #     label.setObjectName(host)
        #     self.vbox.addWidget(label)

        i = 1


        for n in a_list:
            globals()["w" + str(i)] =  QLabel(n)

            list_clear_lab.append(globals()["w" + str(i)])
            link= "https://translate.google.com/?pli=1&sl=en&tl=ar&text=" + globals()["w" + str(i)].text()

            globals()["w" + str(i)].mousePressEvent = lambda e, link=link: callback_pyqt5(link)
            self.vbox.addWidget(globals()["w" + str(i)])
            #del globals()["w" + str(i)]
            #print(globals()["w" + str(i)])

            i += 1

        for name in list_clear_lab: #deleted
           name.deleteLater()



        #list_clear_lab.clear()
        # for name in list_clear_lab:  # deleted
        #     name.setText("Hello")
        #print(len(list_clear_lab))

        #self.label = QLabel('This is label')

        # for i in range(1,50):
        #     label = QLabel('')
        #self.label.move(self.label.x(), self.label.y() + 25)

        self.widget.setLayout(self.vbox)

        # Scroll Area Properties
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        # self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.widget)

        self.setCentralWidget(self.scroll)
        """

    def initUI(self):
        self.setWindowTitle("Lil Snippy")
        self.setWindowIcon(QtGui.QIcon("assets/lilSnippyIcon.png"))
        self.setGeometry(400, 300, 640, 300)

        # QApplication.setOverrideCursor(Qt.WaitCursor)

    def topMenu(self):
        menubar = self.menuBar()

        fileMenu = menubar.addMenu("File")
        saveAct = QtWidgets.QAction(QtGui.QIcon("assets/saveIcon.png"), "Save", self)
        saveAsAct = QtWidgets.QAction(
            QtGui.QIcon("assets/saveAsIcon.png"), "Save As", self
        )

        modeMenu = menubar.addMenu("Mode")
        snipAct = QtWidgets.QAction(QtGui.QIcon("assets/cameraIcon.png"), "Snip", self)
        snipAct.setShortcut(QtGui.QKeySequence("F1"))
        snipAct.triggered.connect(self.activateSnipping)
        videoAct = QtWidgets.QAction(QtGui.QIcon("assets/videoIcon.png"), "Video", self)
        videoAct.setShortcut("F2")
        soundAct = QtWidgets.QAction(QtGui.QIcon("assets/audioIcon.png"), "Sound", self)
        soundAct.setShortcut("F3")
        autoAct = QtWidgets.QAction(
            QtGui.QIcon("assets/automationIcon.png"), "Automation", self
        )
        autoAct.setShortcut("F4")

        optionsMenu = menubar.addMenu("Options")

        helpMenu = menubar.addMenu("Help")
        helpAct = QtWidgets.QAction(QtGui.QIcon("assets/helpIcon.png"), "Help", self)
        aboutAct = QtWidgets.QAction(QtGui.QIcon("assets/aboutIcon.png"), "About", self)

        fileMenu.addAction(saveAct)
        fileMenu.addAction(saveAsAct)
        modeMenu.addAction(snipAct)
        modeMenu.addAction(videoAct)
        modeMenu.addAction(soundAct)
        modeMenu.addAction(autoAct)
        helpMenu.addAction(helpAct)
        helpMenu.addAction(aboutAct)

        self.snipper = SnippingWidget()
        self.snipper.closed.connect(self.on_closed)
        self.snipper.change_text.connect(self.set_text)

    def activateSnipping(self):
        # self.textbox.setText("")
        # self.textbox.setText("active snip")
        self.snipper.showFullScreen()
        QApplication.setOverrideCursor(QtCore.Qt.CrossCursor)
        self.hide()

    def on_closed(self):
        self.show()


class SnippingWidget(QtWidgets.QMainWindow):
    closed = QtCore.pyqtSignal()
    change_text = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super(SnippingWidget, self).__init__(parent)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setStyleSheet("background:transparent;")
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)

        self.outsideSquareColor = "red"
        self.squareThickness = 2

        self.start_point = QtCore.QPoint()
        self.end_point = QtCore.QPoint()

    def mousePressEvent(self, event):
        self.start_point = event.pos()
        self.end_point = event.pos()
        self.update()

    def mouseMoveEvent(self, event):
        self.end_point = event.pos()
        self.update()

    def mouseReleaseEvent(self, QMouseEvent):
        r = QtCore.QRect(self.start_point, self.end_point).normalized()
        self.hide()
        img = ImageGrab.grab(bbox=r.getCoords())
        # img.save("snips/testImage.png")
        output = BytesIO()
        img.save(output, format="JPEG")
        """
        # np array
        x = np.arange(28 * 28).reshape(28, 28)
        print(x.shape)

        np_bytes = BytesIO()
        np.save(np_bytes, x, allow_pickle=True)

        np_bytes = np_bytes.getvalue()
        print(type(np_bytes))

        load_bytes = BytesIO(np_bytes)
        loaded_np = np.load(load_bytes, allow_pickle=True)
        print(loaded_np.shape)


        # output.save(output, x, allow_pickle=True)
        # np_bytes_output = output.getvalue()
        # print(type(np_bytes_output))

        # np array
        """
        try:
            ii = Image.open(output)
            open_cv_image = np.array(ii)
            # Convert RGB to BGR
            open_cv_image = open_cv_image[:, :, ::-1].copy()  # <class 'numpy.ndarray'>

            imbw, imwb = return_image_to_pytesseract(open_cv_image,
                                                     True)  # Image.fromarray(np.uint8(cm.gist_earth(invert) * 255))
            # imbw.show() #need this to pytesseract
            ii.save("D:/A_a_SEU/mid/1.jpg")
            #D:\A_a_SEU\mid
        except Exception as e:
            print("read opencv from PIL.Image: ", e)
        """
        try:
            # file_bytes = np.asarray(bytearray(output.read()), dtype=np.uint8)
            # im = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            #array = np.asarray(bytearray(output.read()), dtype=np.uint8)
            pass
            # im = cv2.imdecode(np.frombuffer(output.read(), np.uint8), 1)
            # cv2.imshow("img", img)
            # cv2.waitKey(0)

            #image = get_opencv_img_from_buffer(self,output) #cv2.imdecode(array, cv2.IMREAD_COLOR)

            #print(str(type(im)))

            #cv2.imshow('Image', image)
        except Exception as e:
            print("this array numpay: ",e)
        try:
            pass
            #img.save(image, format="JPEG")
        except Exception as e:
            print("image grab.save: " ,e)
        #img.save(output, format="JPEG")
        """

        try:
            text_ocr = pytesseract.image_to_string(imbw, lang='eng')  # lang='ara+eng')
            final = filter_text_in_one_line(self, text_ocr)
            print("final: " + "\n" + final)
        except Exception as e:
            print("pytesseract mouse releasse print final text: ", e)

        # QApplication.processEvents()
        QApplication.restoreOverrideCursor()
        self.closed.emit()
        self.change_text.emit(final)
        self.start_point = QtCore.QPoint()
        self.end_point = QtCore.QPoint()

    def paintEvent(self, event):
        trans = QtGui.QColor(22, 100, 233)
        r = QtCore.QRectF(self.start_point, self.end_point).normalized()
        qp = QtGui.QPainter(self)
        trans.setAlphaF(0.2)
        qp.setBrush(trans)
        outer = QtGui.QPainterPath()
        outer.addRect(QtCore.QRectF(self.rect()))
        inner = QtGui.QPainterPath()
        inner.addRect(r)
        r_path = outer - inner
        qp.drawPath(r_path)
        qp.setPen(
            QtGui.QPen(QtGui.QColor(self.outsideSquareColor), self.squareThickness)
        )
        trans.setAlphaF(0)
        qp.setBrush(trans)
        qp.drawRect(r)




def capture_video_specific_window():
    import numpy as np
    import cv2
    from PIL import ImageGrab
    import win32api
    # import winGuiAuto
    import win32gui
    import win32con

    cap = cv2.VideoCapture(0)

    # Capture the window frame by frame
    image_list = []
    for _ in range(70):
        ret, frame = cap.read()
        cv2.imshow('SCORE', frame)
        cv2.waitKey(1)
        hwnd = win32gui.FindWindow(None, r"Live Caption")
        if win32gui.IsWindowVisible(hwnd):
            print("visable")
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                  win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
            rect = win32gui.GetWindowPlacement(hwnd)[-1]
            # (510, 248, 1213, 582)
            rr = win32gui.GetWindowPlacement(hwnd)
            print("rect: " + str(rect))
            print("rr: " + str(rr))
            image = ImageGrab.grab(rect)
            image.show()
            image_list.append(image)
        else:
            print("not visable")
    height, width, channel = np.array(image).shape
    cap.release()
    cv2.destroyAllWindows()

    out = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 5, (width, height))

    for images in image_list:
        out.write(cv2.cvtColor(np.array(images), cv2.COLOR_BGR2RGB))
    out.release()


def capture_without_word_Live_Caption():
    hwnd = win32gui.FindWindow(None, r"Live Caption")
    if win32gui.IsWindowVisible(hwnd):
        print("visable")
        QThread.sleep(1)
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                              win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
        rect = win32gui.GetWindowPlacement(hwnd)[-1]
        n1 = rect[0] + 200  # left
        n2 = rect[1] + 185  # top
        n3 = rect[2] + 305  # right
        n4 = rect[3] + 140  # bottom
        t = (n1, n2, n3, n4)
        print(t)
        # (510, 248, 1213, 582)
        # rr = win32gui.GetWindowPlacement(hwnd)
        # print("rect: " + str(rect))
        # print("rr: " + str(rr))
        image = ImageGrab.grab(t)
        image.show("hh")
        # image_nd = convert_from_image_to_cv2(image)
        pil_image = image.convert('RGB')

        open_cv_image = numpy.array(pil_image)
        # Convert RGB to BGR
        image_nd = open_cv_image[:, :, ::-1].copy()
        #cv2.imshow("Image",image_nd)
        #cv2.waitKey(0)


if __name__ == "__main__":
    # from moviepy.editor import *
    import speech_recognition as sr
    import sys
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import pyplot as PLT
    from PIL import Image
    from io import StringIO
    import PIL
    import matplotlib
    from matplotlib import cm
    import numpy

    
    print("matploit version: ", matplotlib._get_version())
    print("opencv version: ", cv2.__version__)

    try:

        #capture_without_word_Live_Caption()

        app = QtWidgets.QApplication(sys.argv)
        application = App()

        application.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(traceback.format_exc())
        print("Ex main", e)




# try:
#     do_stuff()
# except Exception:
#
#     # or
#     print(sys.exc_info()[2])
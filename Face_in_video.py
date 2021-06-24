import cv2
import numpy as np
import face_recognition
import os
import csv
import time
import PySimpleGUI as sg

class SearchPepole():
    def __init__(self, folderOfPerson="", sizeD=0.25, toleranceD=0.55,folderName=''):
        self.folderOfPerson = folderOfPerson
        self.sizeD = sizeD
        self.toleranceD = toleranceD
        self.folderName = folderName

    def createImageClass(self):
        self.images = []
        self.classNames = []
        self.myList = os.listdir(self.folderOfPerson)
        print(self.myList)
        for cl in self.myList:
            if cl[-4:] == '.jpg':
                currentImg = cv2.imread(f'{self.folderOfPerson}/{cl}')
                # print(currentImg)
                # print(cl)
                self.images.append(currentImg)
                self.classNames.append(os.path.splitext(cl)[0])
        print(self.classNames)
        return self.images, self.classNames

    def findEncodings(self, images):
        encodeList = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            print(f'Encode: {encode}')
            encodeList.append(encode)
        return encodeList

    def markAttendance(self, name, findTime, img):

        with open(f'{self.folderName}/FindList.csv', 'r+') as f:
            myDataList = f.readlines()
            nameList = []
            # print(myDataList)
            for line in myDataList:
                entry = line.split(',')
                nameList.append(entry[0])
            if name not in nameList:
                # now = datetime.now()
                # dtString = now.strftime('%H:%M:%S')

                f.writelines(f'\n{name},{findTime}')

    def timeConvert(self, sec):
        mins = sec // 60
        sec = sec % 60
        hours = mins // 60
        mins = mins % 60
        resultTime = "{0}:{1}:{2}".format(int(hours), int(mins), int(sec))
        return resultTime


    def findPepoleInVideo(self,video):

        images, classNames = self.createImageClass()
        encodeListKnown = self.findEncodings(images)

        cap = cv2.VideoCapture(video)
        cap.set(cv2.CAP_PROP_FPS, 30)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = frame_count / self.fps
        print(self.duration)

        startTime = time.time()
        cunt = 0
        while True:

            # currentNum = currentNum + 1
            # print(currentNum)
            fontSizeOver = 0.7
            hInterval = 40
            fontSize = 0.8
            success, img = cap.read()
            if cap.isOpened():
                cunt += 1
                hP, wP = img.shape[:2]
                print(hP, wP)
                if hP >=1000:
                    fontSize  = 0.8
                    fontSizeOver = 0.7
                    hInterval = 40
                elif hP <=1000 and hP > 800:
                    fontSize = 0.6
                    fontSizeOver = 0.5
                    hInterval = 30
                elif hP <=800 and hP > 600:
                    fontSize = 1
                elif hP <= 600:
                    fontSize = 0.5
                    fontSizeOver = 0.4
                    hInterval = 20
                else:
                    fontSize = 0.5
                    fontSizeOver = 0.4
                    hInterval = 20

                print(f'Frame: {cunt}')
                barr = sg.one_line_progress_meter('Scan Camera!', cunt, frame_count, '-key-', orientation='H')
                # print(barr)
                if barr == False:
                    break

                imgS = cv2.resize(img, (0, 0), None, self.sizeD, self.sizeD)
                imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)


                facesCurFrame = face_recognition.face_locations(imgS)
                encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
                forcunt = 0
                for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                    matches = face_recognition.compare_faces(encodeListKnown, encodeFace, tolerance=self.toleranceD)
                    faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                    # print(faceDis)
                    matchIndex = np.argmin(faceDis)
                    print(f'faceDis: {faceDis}')
                    print(f'matchIndex: {matchIndex}')
                    print(f'faceDis[matchIndex]: {faceDis[matchIndex]}')

                    print(f'forcunt {forcunt}')
                    matchPer = (1 - faceDis[matchIndex]) * 100
                    if matches[matchIndex]:
                        name = classNames[matchIndex].upper()
                        # print(name)
                        findTime = time.time()

                        y1, x2, y2, x1 = faceLoc
                        y1, x2, y2, x1 = int(y1 * (1/self.sizeD)), int(x2 * (1/self.sizeD)), int(y2 * (1/self.sizeD)), int(x1 * (1/self.sizeD))
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
                        cv2.rectangle(img, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), (0, 0, 200), 2)
                        cv2.putText(img, f'{matchPer:.2f}%', (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, fontSizeOver,
                                    (255, 255, 255), 1)
                        cv2.putText(img, name, (x1, y1 - hInterval), cv2.FONT_HERSHEY_COMPLEX, fontSizeOver,
                                    (255, 255, 255), 1)
                        # cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                        timeFrameN = self.timeConvert(int(cunt / self.fps))
                        videoTime = findTime - startTime
                        videoTimeFind = self.timeConvert(videoTime)
                        cv2.putText(img, name, (20, hInterval + (forcunt * 100)), cv2.FONT_HERSHEY_COMPLEX, fontSize, (200, 200, 200), 1)
                        cv2.putText(img, f'{matchPer:.2f}%', (20, (hInterval * 2 ) + (forcunt * 100)), cv2.FONT_HERSHEY_COMPLEX, fontSize, (200, 200, 200), 1)
                        cv2.putText(img, f'Find Time{str(videoTimeFind)}', (20, (hInterval * 3) + (forcunt * 100)), cv2.FONT_HERSHEY_COMPLEX, fontSize, (200, 200, 200), 1)
                        cv2.putText(img, f'Time in Video ** {str(timeFrameN)} **', (20, (hInterval * 4) + (forcunt * 100)), cv2.FONT_HERSHEY_COMPLEX, fontSize , (200, 200, 200), 2)
                        cv2.putText(img, str(cunt), (20, (hInterval * 5) + (forcunt * 100)), cv2.FONT_HERSHEY_COMPLEX, fontSize, (255, 255, 255), 1)
                        print(name)
                        forcunt += 1

                        print(videoTimeFind)
                        cv2.imwrite(f'{self.folderName}/FoundPeople/{name} {str(cunt)}.jpg', img)
                        self.markAttendance(name, videoTimeFind, img)
            else:
                continue


def main():
    layout = [[sg.Frame('Select Files',
                        layout=[[sg.T("Size Of Video")],
                                [sg.Slider(range=(10, 200), orientation='h', size=(34, 20), default_value=25,
                                           key='_sizeOfVideo_', enable_events=True)],
                                [sg.T("Tolerance")],
                                [sg.Slider(range=(10, 90), orientation='h', size=(34, 20), default_value=65,
                                           key='_tolerance_', enable_events=True)],
                                [sg.T("")], [sg.Text("Choose a Video file:  "), sg.Input(readonly=True),
                                             sg.FileBrowse(file_types=[("Video Files", "*.mp4"),
                                                                       ("Video Files", "*.mkv"),
                                                                       ("Video Files", "*.avi")], key='_file_')],
                                [sg.T("")], [sg.Text("Choose Image folder: "), sg.Input(readonly=True),
                                sg.FolderBrowse(key="_IN_")],
                                [sg.Button('Ok', key='_OK_', size=(10, 2), button_color='Green', pad=(10, 10)),
                                sg.Button('Exit', key='_Exit_', size=(10, 2), button_color='red', pad=(10, 10))]
                                ], element_justification='c', border_width=1, vertical_alignment='c')
               ]]

    windowSelect = sg.Window('My File Browser', layout)

    while True:


        event, values = windowSelect.read()


        print(event)
        print(values['_sizeOfVideo_'])
        sizeOfVideo = float(int(values['_sizeOfVideo_']) / 100)
        print(sizeOfVideo)
        tolerance = float(int(values['_tolerance_']) / 100)
        print(tolerance)

        if event in (None, '_Exit_'):
            exit()
        elif event == '_OK_':
            if len(values['_file_']) != 0 and len(values['_IN_']) != 0:
                filename = values['_file_']
                foldername = values['_IN_']
                print(filename)
                print(foldername)
                print(os.path.splitext(filename))
                print(foldername)
                # path = os.path.dirname(foldername)
                dirName = f'{foldername}/FoundPeople'
                if not os.path.exists(dirName):
                    os.makedirs(dirName)
                    print("Directory ", dirName, " Created ")

                row_list = [
                    ["Name", "Time and Date"]
                ]

                with open(f'{foldername}/FindList.csv', 'w', newline='') as file:
                    writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC,
                                        delimiter=',')
                    writer.writerows(row_list)

                # windowSelect.close()
                detector = SearchPepole(folderOfPerson=foldername, sizeD=sizeOfVideo, toleranceD=tolerance,
                                        folderName=foldername)

                detector.findPepoleInVideo(filename)



            elif len(values['_file_']) == 0 and len(values['_IN_']) != 0:
                sg.popup('Please Select a Video for Searching',
                         no_titlebar=True, keep_on_top=True, background_color='red')

            elif len(values['_file_']) != 0 and len(values['_IN_']) == 0:
                sg.popup('Please Select Images Folder for comparison',
                         no_titlebar=True, keep_on_top=True, background_color='red')
            else:
                strError = "Please Select Images Folder for comparison\nPlease Select a Video for Searching"
                sg.popup(strError, no_titlebar=True, keep_on_top=True, background_color='red')


if __name__ == "__main__":
    main()

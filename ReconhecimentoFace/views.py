from django.shortcuts import render, redirect
from django.conf import settings
from django.contrib import messages
from django.contrib.auth.models import User
from .forms import UserSelection

import cv2
import numpy as np
import os
from PIL import Image

BASE_DIR = getattr(settings, 'BASE_DIR')

def index(request):
    context = {'forms': UserSelection }
    return render(request, 'index.html', context)

def create_dataset(request):
    if request.method == "POST":
        face_id = int(request.POST['selected_user'])
        #print("Face ID->", face_id, type(face_id))

        cam = cv2.VideoCapture(0)
        cam.set(3, 640)  # definir a largura do vídeo
        cam.set(4, 480)  # definir altura do vídeo
        face_detector = cv2.CascadeClassifier(BASE_DIR + '/treinamento/haarcascade_frontalface_default.xml')  # Para cada pessoa, insira um ID de rosto numérico
        print("[INFO] Inicializando a captura de rosto. Olhe para câmera e espere ...")  # Inicialize a contagem de faces de amostragem individual
        count = 0
        while (True):
            ret, img = cam.read()
            # img = cv2.flip(img, -1) # virar imagem de vídeo verticalmente
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)

            # Ignore o processo se vários rostos forem detectados:
            if len(faces) == 1:
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    count += 1
                    # Salve a imagem capturada na pasta de conjuntos de dados e atribuir ID para o Usuário
                    cv2.imwrite(BASE_DIR+"/treinamento/dataset/User." + str(face_id) + '.' +
                                str(count) + ".jpg", gray[y:y + h, x:x + w])
                    cv2.waitKey(250)

                cv2.imshow('Face', img)
                k = cv2.waitKey(1) & 0xff  # Pressione 'ESC' para sair do vídeo
                if k == 27:
                    break
                elif count >= 30:  # Pegar 30 amostras de rosto e parar o vídeo
                    break  
                print(count)
            else:
                print("\n Multiplas faces detectadas")

        print("\n [INFO] Fechando o programa ... ")
        cam.release()
        cv2.destroyAllWindows()

        messages.success(request, 'Foto Cadastrada com Sucesso !')

    else:
        print("Metodo GET.")

    return redirect('/')

def detectar(request):
    faceDetect = cv2.CascadeClassifier(BASE_DIR + '/treinamento/haarcascade_frontalface_default.xml')

    cam = cv2.VideoCapture(0) 
    # criando reconhecedor
    rec = cv2.face.LBPHFaceRecognizer_create();
    # carregando os dados de treinamento
    rec.read(BASE_DIR + '/treinamento/reconhecedor/treinar.yml')
    getId = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    userId = 0
    while (True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            getId, conf = rec.predict(gray[y:y + h, x:x + w])  # Isso irá prever o id do rosto
            print(getId, conf)
            confidence = "  {0}%".format(round(100 - conf))
            # print conf;
            if conf < 35:
                try:
                    user = User.objects.get(id=getId)
                except  User.DoesNotExist:
                    pass

                print("User Name", user.username)

                userId = getId
                if user.username:
                    cv2.putText(img, user.username, (x+5, y+h-10), font, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(img, "Detectada", (x, y + h), font, 1, (0, 255, 0), 2)
            else:
                cv2.putText(img, "Face Desconhecida", (x, y + h), font, 1, (0, 0, 255), 2)

            cv2.putText(img, str(confidence), (x + 5, y - 5), font, 1, (255, 255, 0), 1)
            # Imprimindo esse número abaixo do rosto
            # @Prams cam imagem, id, localização, estilo de fonte, cor, traçado

        cv2.imshow("Face", img)
        if (cv2.waitKey(1) == ord('q')):
            break
        #elif (userId != 0):
        #    cv2.waitKey(1000)
        #    cam.release()
        #    cv2.destroyAllWindows()
        #    return redirect('/records/details/' + str(userId))

    cam.release()
    cv2.destroyAllWindows()
    return redirect('/')

def treinar(request):
    '''
        Em treinar.py, temos que obter todas as amostras da pasta do conjunto de dados,
        para o treinador reconhecer qual número de identificação é para qual rosto.
        para isso precisamos extrair todo o caminho relativo
        ou seja, dataset/user.1.1.jpg, dataset/user.1.2.jpg, dataset/user.1.3.jpg.
    '''


    # caminho para banco de dados de imagens de rosto
    path = BASE_DIR + '/treinamento/dataset'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(BASE_DIR+"/treinamento/haarcascade_frontalface_default.xml");  # função para obter as imagens e os dados do rótulo

    def getImagesAndLabels(path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        ids = []
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L')  # escala de cinza
            img_numpy = np.array(PIL_img, 'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            #faces = detector.detectMultiScale(img_numpy)
            #for (x, y, w, h) in faces:
            #    faceSamples.append(img_numpy[y:y + h, x:x + w])
            #    ids.append(id)
            faceSamples.append(img_numpy)
            ids.append(id)
            # print ID
            cv2.imshow("treinando as fotos", img_numpy)
            cv2.waitKey(10)
        return np.array(faceSamples), np.array(ids)
        #return faceSamples, ids

    print("[INFO] Treinando faces. Vai demorar alguns segundos. Espere...")

    faces, ids = getImagesAndLabels(path)
    recognizer.train(faces, ids)  # Salvar o modelo em treinamento/treinar.yml
    recognizer.save(BASE_DIR+'/treinamento/reconhecedor/treinar.yml')  # Treinar o número de faces  e finalize o programa
    print("[INFO] {0} faces treinada. Fechar o Sistema".format(len(np.unique(ids))))
    cv2.destroyAllWindows()
    messages.success(request, "{0} faces treinada com sucesso".format(len(np.unique(ids))) )

    return redirect('/')
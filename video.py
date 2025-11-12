import cv2
import numpy as np
from abc import ABC, abstractmethod  # Para crear clases base abstractas (filtros)
import time
import argparse
import sys

class CaptureManager:
    """
    Gestiona la captura y procesamiento de frames de un objeto de captura de video.
    """
    
    def __init__(self, capture, preview_window_manager=None):
        """
        Inicializa el gestor de captura.

        Argumentos:
            capture: El objeto cv2.VideoCapture (fuente de video).
            preview_window_manager: (Opcional) Un WindowManager para mostrar los frames.
        """
        self.preview_window_manager = preview_window_manager
        self._capture = capture
        self._channel = 0  # Canal de captura (más relevante para cámaras)
        self._entered_frame = False  # Flag para rastrear si se ha llamado a enter_frame
        self._frame = None  # Almacena el frame actual
        self._image_filename = None  # Nombre de archivo si se solicita una captura de pantalla
        self._video_filename = None
        self._video_encoding = None
        self._video_writer = None
        
        # Variables para estimar los FPS (fotogramas por segundo)
        self._start_time = None
        self._frames_elapsed = 0
        self._fps_estimate = None
    
    @property
    def channel(self):
        """Propiedad para obtener el canal de captura."""
        return self._channel
    
    @channel.setter
    def channel(self, value):
        """Propiedad para establecer el canal de captura."""
        if self._channel != value:
            self._channel = value
            self._frame = None
    
    @property
    def frame(self):
        """
        Propiedad para obtener el frame actual.
        Si enter_frame() fue llamado, recupera el frame del buffer de captura.
        """
        if self._entered_frame and self._frame is None:
            # retrieve() decodifica y devuelve el frame "agarrado" por grab()
            _, self._frame = self._capture.retrieve(self._frame, self.channel)
        return self._frame
    
    @property
    def is_writing_image(self):
        """Devuelve True si se está en proceso de escribir una imagen."""
        return self._image_filename is not None
    
    @property
    def is_writing_video(self):
        """Devuelve True si se está en proceso de escribir un video."""
        return self._video_filename is not None
    
    def enter_frame(self):
        """
        Captura (agarra) el siguiente frame del video.
        Este es el primer paso de la captura en dos pasos (grab/retrieve).
        """
        # Asegurarse de que no llamemos a enter_frame() dos veces sin un exit_frame()
        assert not self._entered_frame, \
            'enter_frame() previo no tuvo un exit_frame() correspondiente'
        
        if self._capture is not None:
            # grab() toma el frame de la fuente de video
            self._entered_frame = self._capture.grab()
    
    def exit_frame(self):
        """
        Procesa el frame capturado: lo muestra, lo escribe en archivo (si se solicita)
        y libera los recursos del frame.
        """
        
        # Si no hay frame (p.ej., fin del video) o no se capturó nada, salir.
        if self.frame is None:
            self._entered_frame = False
            return
        
        # Actualizar la estimación de FPS
        if self._frames_elapsed == 0:
            self._start_time = time.time()
        else:
            time_elapsed = time.time() - self._start_time
            self._fps_estimate = self._frames_elapsed / time_elapsed
        self._frames_elapsed += 1
        
        # Mostrar el frame en la ventana, si hay un gestor de ventanas
        if self.preview_window_manager is not None:
            self.preview_window_manager.show(self._frame)
        
        # Escribir el frame a un archivo de imagen (captura de pantalla) si se solicitó
        if self.is_writing_image:
            cv2.imwrite(self._image_filename, self._frame)
            self._image_filename = None  # Resetear el flag
        
        # Liberar el frame actual y resetear el flag de entrada
        self._frame = None
        self._entered_frame = False
    
    def write_image(self, filename):
        """
        Establece el nombre de archivo para guardar el *próximo* frame procesado
        en exit_frame().
        """
        self._image_filename = filename

    def release(self):
        """Libera el objeto de captura de video (cierra el archivo/cámara)."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class WindowManager:
    """
    Gestiona una ventana de OpenCV y los eventos de teclado/mouse asociados.
    """
    
    def __init__(self, window_name, keypressed_callback=None):
        """
        Inicializa el gestor de ventanas.

        Argumentos:
            window_name: El nombre que aparecerá en la barra de título de la ventana.
            keypressed_callback: (Opcional) Función a llamar cuando se presiona una tecla.
        """
        self.keypressed_callback = keypressed_callback
        self._window_name = window_name
        self._is_window_created = False
    
    @property
    def is_window_created(self):
        """Devuelve True si la ventana ha sido creada."""
        return self._is_window_created
    
    def create_window(self):
        """Crea la ventana de OpenCV."""
        cv2.namedWindow(self._window_name)
        self._is_window_created = True
    
    def show(self, frame):
        """Muestra un frame en la ventana."""
        cv2.imshow(self._window_name, frame)
    
    def destroy_window(self):
        """Destruye la ventana de OpenCV y resetea el flag."""
        cv2.destroyWindow(self._window_name)
        self._is_window_created = False
    
    def process_events(self, keycode):
        """
        Procesa los eventos de teclado.
        El keycode es pasado desde el bucle principal (que llama a cv2.waitKey).
        """
        if self.keypressed_callback is not None and keycode != -1:
            # Si se presionó una tecla, llama a la función callback
            self.keypressed_callback(keycode)


class Filter(ABC):
    """
    Clase base abstracta (ABC) para todos los filtros.
    Define la interfaz que todos los filtros deben implementar.
    """
    
    @abstractmethod
    def apply(self, frame):
        """
        Aplica el efecto del filtro a un frame.
        Debe ser implementado por las subclases.
        """
        pass
    
    @abstractmethod
    def get_name(self):
        """
        Devuelve el nombre del filtro para mostrar en la GUI.
        Debe ser implementado por las subclases.
        """
        pass


class NoFilter(Filter):
    """Un filtro que no hace nada; devuelve el frame original."""
    def apply(self, frame):
        return frame
    def get_name(self):
        return "Original"

class BlurFilter(Filter):
    """Aplica un filtro de desenfoque Gaussiano."""
    def apply(self, frame):
        return cv2.GaussianBlur(frame, (15, 15), 0)
    def get_name(self):
        return "Desenfoque (Blur)"

class SharpenFilter(Filter):
    """Aplica un filtro de enfoque (sharpen) usando un kernel."""
    def apply(self, frame):
        # Kernel simple para enfocar
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        return cv2.filter2D(frame, -1, kernel)
    def get_name(self):
        return "Enfoque (Sharpen)"

class EdgeDetectionFilter(Filter):
    """Aplica detección de bordes usando el algoritmo Canny."""
    def apply(self, frame):
        # Convertir a escala de grises primero, ya que Canny lo requiere
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        # Convertir de nuevo a BGR (3 canales) para que sea compatible
        # con otros filtros y el video original.
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    def get_name(self):
        return "Deteccion de Bordes"

class GrayscaleFilter(Filter):
    """Convierte el frame a escala de grises."""
    def apply(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Convertir de nuevo a BGR para mantener la consistencia de 3 canales
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    def get_name(self):
        return "Escala de Grises"

class ChannelSplitFilter(Filter):
    """Divide y muestra los canales BGR (Azul, Verde, Rojo) lado a lado."""
    def apply(self, frame):
        b, g, r = cv2.split(frame) # Dividir en canales
        # Crear una matriz de ceros con las mismas dimensiones
        zeros = np.zeros(b.shape, dtype=np.uint8)
        
        # Combinar cada canal con ceros para aislar el color
        blue_channel = cv2.merge([b, zeros, zeros])
        green_channel = cv2.merge([zeros, g, zeros])
        red_channel = cv2.merge([zeros, zeros, r])
        
        # Concatenar las imágenes horizontalmente (temporalmente)
        h, w = frame.shape[:2]
        result = np.zeros((h, w * 3, 3), dtype=np.uint8)
        result[:, 0:w] = blue_channel
        result[:, w:w*2] = green_channel
        result[:, w*2:w*3] = red_channel
        
        # Redimensionar de nuevo al tamaño original para visualización
        result = cv2.resize(result, (w, h))
        return result
    def get_name(self):
        return "Canales BGR"

class CartoonizeFilter(Filter):
    """Aplica un efecto de "caricatura" al frame."""
    def apply(self, frame):
        # Aplicar filtro bilateral para suavizar colores pero mantener bordes
        img_color = cv2.bilateralFilter(frame, 9, 250, 250)
        # Convertir a escala de grises para la detección de bordes
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Aplicar desenfoque de mediana para reducir ruido
        gray = cv2.medianBlur(gray, 7)
        # Usar umbral adaptativo para obtener los bordes (como un dibujo)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
        # Convertir bordes a 3 canales
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        # Combinar la imagen de color suavizada con los bordes usando AND bitwise
        cartoon = cv2.bitwise_and(img_color, edges)
        return cartoon
    def get_name(self):
        return "Caricatura (Cartoonize)"
    
class MotionBlurFilter(Filter):
    """Aplica un desenfoque de movimiento (motion blur) horizontal simple."""
    def apply(self, frame):
        kernel_size = 15
        # Crear un kernel que solo tenga valores en la fila central
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        # Normalizar el kernel
        kernel = kernel / kernel_size
        return cv2.filter2D(frame, -1, kernel)

    def get_name(self):
        return "Desenfoque de Movimiento"


class EmbossFilter(Filter):
    """Aplica un efecto de relieve (emboss)."""
    def apply(self, frame):
        # Kernel de relieve
        kernel = np.array([[-2, -1, 0],
                           [-1,  1, 1],
                           [ 0,  1, 2]])
        return cv2.filter2D(frame, -1, kernel)

    def get_name(self):
        return "Relieve (Emboss)"


class ErosionFilter(Filter):
    """Aplica erosión para reducir (erosionar) las áreas brillantes."""
    def apply(self, frame):
        # Kernel cuadrado de 5x5
        kernel = np.ones((5, 5), np.uint8)
        return cv2.erode(frame, kernel, iterations=1)

    def get_name(self):
        return "Erosion"


class DilationFilter(Filter):
    """Aplica dilatación para expandir (dilatar) las áreas brillantes."""
    def apply(self, frame):
        # Kernel cuadrado de 5x5
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(frame, kernel, iterations=1)

    def get_name(self):
        return "Dilatacion"


class VignetteFilter(Filter):
    """Aplica un efecto de viñeta, oscureciendo las esquinas."""
    def apply(self, frame):
        rows, cols = frame.shape[:2]
        # Generar una máscara Gaussiana 2D
        kernel_x = cv2.getGaussianKernel(cols, cols / 2)
        kernel_y = cv2.getGaussianKernel(rows, rows / 2)
        kernel = kernel_y * kernel_x.T
        
        # Normalizar la máscara y crear 3 canales para el color
        mask = kernel / kernel.max()
        mask_3ch = np.dstack([mask] * 3)
        
        # Aplicar la máscara multiplicando
        result = frame.copy().astype(float)
        result *= mask_3ch
        return result.astype(np.uint8)

    def get_name(self):
        return "Vignette"


class ContrastEnhanceFilter(Filter):
    """Mejora el contraste local usando CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    def apply(self, frame):
        # Convertir al espacio de color LAB
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Crear un objeto CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        # Aplicar CLAHE solo al canal de Luminosidad (L)
        cl = clahe.apply(l)
        
        # Unir los canales de nuevo
        limg = cv2.merge((cl, a, b))
        # Convertir de nuevo a BGR
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def get_name(self):
        return "Mejora de Contraste"


class MoustacheFilter(Filter):
    """Detecta una nariz y superpone una imagen de bigote debajo."""
    
    def __init__(self):
        """Carga los clasificadores Haar y la imagen del bigote."""
        cascade_path = cv2.data.haarcascades
        self.nose_cascade = cv2.CascadeClassifier(
            cascade_path + 'haarcascade_mcs_nose.xml')
        
        # Cargar la imagen del bigote (con canal alfa para transparencia)
        try:
            self.moustache_img = cv2.imread('livevideo/mustache.png', cv2.IMREAD_UNCHANGED)
            if self.moustache_img is None:
                print("Advertencia: No se encontró mustache.png. Creando placeholder.")
                self.moustache_img = self.create_placeholder_moustache()
        except Exception as e:
            print(f"Error cargando mustache.png: {e}. Creando placeholder.")
            self.moustache_img = self.create_placeholder_moustache()

    def create_placeholder_moustache(self):
        """Crea una imagen de reemplazo si no se encuentra el archivo original."""
        moustache = np.zeros((50, 150, 4), dtype=np.uint8)
        cv2.putText(moustache, 'mustache.png', (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255, 255), 1)
        return moustache
    
    def overlay_image_alpha(self, img, img_overlay, x, y):
        """
        Superpone una imagen (img_overlay) sobre otra (img) en la posición (x, y).
        Maneja la transparencia usando el canal alfa.
        """
        # Obtener dimensiones y coordenadas de la región de interés (ROI)
        y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
        x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])
        
        y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
        x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)
        
        if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
            return
        
        # Recortar la región de interés de la imagen principal y la superposición
        img_crop = img[y1:y2, x1:x2]
        img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
        
        if img_overlay_crop.shape[2] == 4: # Verificar si hay canal alfa
            # Calcular el canal alfa y su inverso
            alpha = img_overlay_crop[:, :, 3] / 255.0
            alpha_inv = 1.0 - alpha
            
            # Aplicar la mezcla alfa para cada canal de color (B, G, R)
            for c in range(3):
                img_crop[:, :, c] = (alpha * img_overlay_crop[:, :, c] +
                                    alpha_inv * img_crop[:, :, c])
        else:
            # Si no hay alfa, simplemente copiar la imagen
            img_crop[:] = img_overlay_crop

    def apply(self, frame):
        """Aplica el filtro de bigote."""
        result = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar narices en la imagen
        noses = self.nose_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(25, 25))
        
        for (nx, ny, nw, nh) in noses:
            # Calcular dimensiones del bigote basado en el ancho de la nariz
            moustache_width = int(nw * 1.8)
            aspect_ratio = self.moustache_img.shape[0] / self.moustache_img.shape[1]
            moustache_height = int(moustache_width * aspect_ratio)
            
            # Redimensionar el bigote
            moustache_resized = cv2.resize(
                self.moustache_img, (moustache_width, moustache_height))
            
            # Calcular posición para poner el bigote justo debajo de la nariz
            mx = nx - int((moustache_width - nw) / 2)
            my = ny + int(nh * 0.05) # Ligeramente superpuesto
            
            # Superponer la imagen del bigote
            self.overlay_image_alpha(result, moustache_resized, mx, my)
            
        return result
    
    def get_name(self):
        return "Overlay de Bigote"
    
class FaceDetectionFilter(Filter):
    """Detecta caras, ojos, narices y bocas y dibuja rectángulos alrededor."""
    
    def __init__(self):
        """Carga todos los clasificadores Haar necesarios."""
        cascade_path = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(
            cascade_path + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(
            cascade_path + 'haarcascade_eye.xml')
        self.nose_cascade = cv2.CascadeClassifier(
            cascade_path + 'haarcascade_mcs_nose.xml')
        self.mouth_cascade = cv2.CascadeClassifier(
            cascade_path + 'haarcascade_mcs_mouth.xml')
    
    def apply(self, frame):
        """Aplica el filtro de detección facial."""
        result = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar caras
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            # Dibujar rectángulo de cara (Azul)
            cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(result, 'Cara', (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Crear una Región de Interés (ROI) para la cara
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = result[y:y+h, x:x+w]
            
            # Detectar ojos en la ROI de la cara (Verde)
            eyes = self.eye_cascade.detectMultiScale(
                roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            # Detectar nariz en la ROI de la cara (Cian)
            noses = self.nose_cascade.detectMultiScale(
                roi_gray, scaleFactor=1.4, minNeighbors=5)
            for (nx, ny, nw, nh) in noses:
                cv2.rectangle(roi_color, (nx, ny), (nx+nw, ny+nh), (255, 255, 0), 2)
                
            # Detectar boca en la ROI de la cara (Rojo)
            mouths = self.mouth_cascade.detectMultiScale(
                roi_gray, scaleFactor=1.7, minNeighbors=11, minSize=(25, 25))
            for (mx, my, mw, mh) in mouths:
                cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (0, 0, 255), 2)

        return result
    
    def get_name(self):
        return "Deteccion Facial Completa"

class EyeglassesFilter(Filter):
    """Superpone una imagen de lentes sobre los ojos detectados."""
    
    def __init__(self):
        """Carga los clasificadores de cara/ojos y la imagen de lentes."""
        cascade_path = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(
            cascade_path + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(
            cascade_path + 'haarcascade_eye.xml')
        
        # Cargar la imagen de lentes
        try:
            self.glasses_img = cv2.imread('livevideo/glasses.png', cv2.IMREAD_UNCHANGED)
            if self.glasses_img is None:
                print("Advertencia: No se encontró glasses.png. Creando placeholder.")
                self.glasses_img = self.create_placeholder_glasses()
        except:
            print("Error cargando glasses.png. Creando placeholder.")
            self.glasses_img = self.create_placeholder_glasses()
    
    def create_placeholder_glasses(self):
        """Crea una imagen de reemplazo si no se encuentra el archivo original."""
        glasses = np.zeros((100, 300, 4), dtype=np.uint8)
        cv2.putText(glasses, 'glasses.png missing', (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255, 255), 1)
        return glasses
    
    # Re-utilizamos la misma función de 'overlay'
    def overlay_image_alpha(self, img, img_overlay, x, y):
        y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
        x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])
        y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
        x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)
        
        if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
            return
        
        img_crop = img[y1:y2, x1:x2]
        img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
        
        if img_overlay_crop.shape[2] == 4:
            alpha = img_overlay_crop[:, :, 3] / 255.0
            alpha_inv = 1.0 - alpha
            for c in range(3):
                img_crop[:, :, c] = (alpha * img_overlay_crop[:, :, c] +
                                    alpha_inv * img_crop[:, :, c])
        else:
            img_crop[:] = img_overlay_crop
    
    def apply(self, frame):
        """Aplica el filtro de lentes."""
        result = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar caras
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (fx, fy, fw, fh) in faces:
            # Crear ROI para la cara
            roi_gray = gray[fy:fy+fh, fx:fx+fw]
            # Detectar ojos DENTRO de la cara
            eyes = self.eye_cascade.detectMultiScale(
                roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
            
            # Necesitamos al menos dos ojos
            if len(eyes) >= 2:
                # Ordenar los ojos por su coordenada x (izquierda a derecha)
                eyes = sorted(eyes, key=lambda e: e[0])
                eye1, eye2 = eyes[0], eyes[1]
                
                # Calcular el centro de cada ojo (coordenadas relativas al frame)
                eye1_center = (fx + eye1[0] + eye1[2]//2, fy + eye1[1] + eye1[3]//2)
                eye2_center = (fx + eye2[0] + eye2[2]//2, fy + eye2[1] + eye2[3]//2)
                
                # Calcular la distancia entre los centros de los ojos
                eye_distance = abs(eye2_center[0] - eye1_center[0])
                
                # Evitar división por cero si los ojos están alineados verticalmente
                if eye_distance > 0:
                    # Calcular el ancho de los lentes basado en la distancia
                    glasses_width = int(eye_distance * 2.3)
                    
                    # Mantener la proporción de la imagen de los lentes
                    aspect_ratio = self.glasses_img.shape[0] / self.glasses_img.shape[1]
                    glasses_height = int(glasses_width * aspect_ratio)
                    
                    # Redimensionar los lentes
                    glasses_resized = cv2.resize(self.glasses_img, (glasses_width, glasses_height))
                    
                    # Calcular la posición (esquina superior izquierda) de los lentes
                    # Centrado horizontal y verticalmente entre los dos ojos
                    glasses_x = int((eye1_center[0] + eye2_center[0]) / 2 - glasses_width / 2)
                    glasses_y = int((eye1_center[1] + eye2_center[1]) / 2 - glasses_height / 2)
                    
                    # Superponer la imagen
                    self.overlay_image_alpha(result, glasses_resized, glasses_x, glasses_y)
        
        return result
    
    def get_name(self):
        return "Overlay de Lentes"


class FullFaceMaskFilter(Filter):
    """Superpone una máscara de cara completa sobre las caras detectadas."""
    
    def __init__(self):
        """Carga el clasificador de cara y la imagen de la máscara."""
        cascade_path = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(
            cascade_path + 'haarcascade_frontalface_default.xml')
        
        # Cargar la imagen de la máscara
        try:
            self.mask_img = cv2.imread('livevideo/mask.png', cv2.IMREAD_UNCHANGED)
            if self.mask_img is None:
                print("Advertencia: No se encontró full_mask.png. Creando placeholder.")
                self.mask_img = self.create_placeholder_mask()
        except Exception as e:
            print(f"Error cargando full_mask.png: {e}. Creando placeholder.")
            self.mask_img = self.create_placeholder_mask()
    
    def create_placeholder_mask(self):
        """Crea una imagen de reemplazo si no se encuentra el archivo original."""
        mask = np.zeros((200, 200, 4), dtype=np.uint8)
        cv2.putText(mask, 'full_mask.png', (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255, 255), 1)
        cv2.putText(mask, 'missing', (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255, 255), 1)
        return mask
    
    # Re-utilizamos la misma función de 'overlay'
    def overlay_image_alpha(self, img, img_overlay, x, y):
        y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
        x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])
        y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
        x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)
        
        if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
            return
        
        img_crop = img[y1:y2, x1:x2]
        img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
        
        if img_overlay_crop.shape[2] == 4:
            alpha = img_overlay_crop[:, :, 3] / 255.0
            alpha_inv = 1.0 - alpha
            for c in range(3):
                img_crop[:, :, c] = (alpha * img_overlay_crop[:, :, c] +
                                    alpha_inv * img_crop[:, :, c])
        else:
            img_crop[:] = img_overlay_crop
    
    def apply(self, frame):
        """Aplica el filtro de máscara."""
        result = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar caras
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            # Calcular dimensiones de la máscara (ligeramente más grande que la cara)
            mask_width = int(w * 1.1)
            
            # Mantener la proporción de la máscara
            aspect_ratio = self.mask_img.shape[0] / self.mask_img.shape[1]
            mask_height = int(mask_width * aspect_ratio)
            
            # Redimensionar la máscara
            mask_resized = cv2.resize(self.mask_img, (mask_width, mask_height))
            
            # Calcular la posición para centrar la máscara sobre la cara
            mask_x = x - int((mask_width - w) / 2)
            mask_y = y - int((mask_height - h) / 2)
            
            # Superponer la imagen
            self.overlay_image_alpha(result, mask_resized, mask_x, mask_y)
        
        return result
    
    def get_name(self):
        return "Overlay de Mascara"


class CameoApp:
    """Clase principal de la aplicación que une todo."""
    
    def __init__(self, video_source):
        """
        Inicializa la aplicación.
        
        Argumentos:
            video_source: Ruta al archivo de video.
        """
        self._window_manager = WindowManager('Cameo', self.on_keypress)
        
        # Cargar la fuente de video
        self._capture = cv2.VideoCapture(video_source)
        if not self._capture.isOpened():
            print(f"Error: No se pudo abrir el archivo de video: '{video_source}'")
            sys.exit(1)
        
        # Obtener los FPS originales del video
        self._original_fps = self._capture.get(cv2.CAP_PROP_FPS)
        if self._original_fps <= 0:
            print("Advertencia: No se pudieron obtener FPS válidos. Usando 30 FPS por defecto.")
            self._original_fps = 30
        
        # Esto ralentiza el bucle 'while' para que coincida con los FPS del video.
        self._wait_time_ms = int(1000 / self._original_fps)
        print(f"Video cargado. FPS Original: {self._original_fps:.2f}, Tiempo de espera: {self._wait_time_ms}ms")
            
        # Inicializar el gestor de captura
        self._capture_manager = CaptureManager(
            self._capture, self._window_manager)
        
        # Crear la lista de todos los filtros disponibles
        self._filters = [
            NoFilter(),
            GrayscaleFilter(),
            BlurFilter(),
            SharpenFilter(),
            EdgeDetectionFilter(),
            MotionBlurFilter(),
            EmbossFilter(),
            ErosionFilter(),
            DilationFilter(),
            VignetteFilter(),
            ContrastEnhanceFilter(),
            ChannelSplitFilter(),
            CartoonizeFilter(),
            FaceDetectionFilter(), 
            MoustacheFilter(),
            EyeglassesFilter(),
            FullFaceMaskFilter()
        ]
        self._current_filter_index = 0  # Filtro actual (inicia en 'Original')
        self._screenshot_counter = 0    # Contador para capturas de pantalla
    
    def run(self):
        """Ejecuta el bucle principal de la aplicación."""
        self._window_manager.create_window()
        
        while self._window_manager.is_window_created:
            # Capturar el frame
            self._capture_manager.enter_frame()
            frame = self._capture_manager.frame
            
            if frame is None:
                print("Video finalizado. Reiniciando...")
                # Si 'frame' es None, el video terminó.
                # Regresar el video al fotograma 0.
                self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                # Volver a capturar el primer frame inmediatamente
                self._capture_manager.enter_frame()
                frame = self._capture_manager.frame
                
                # Si sigue siendo None, hay un problema grave
                if frame is None:
                    print("Error: No se pudo leer el frame después de reiniciar. Saliendo.")
                    break
            
            # Aplicar el filtro actual
            current_filter = self._filters[self._current_filter_index]
            filtered_frame = current_filter.apply(frame)
            
            # Dibujar texto de ayuda en el frame
            filter_name = current_filter.get_name()
            cv2.putText(filtered_frame, f'Filtro: {filter_name}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(filtered_frame, 'Q/E: Cambiar Filtro | S: Screenshot | ESC: Salir',
                       (10, filtered_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Asignar el frame filtrado para mostrar
            self._capture_manager._frame = filtered_frame
            
            # Mostrar el frame y procesar guardado de imagen
            self._capture_manager.exit_frame()
            
            # Procesar eventos de teclado
            keycode = cv2.waitKey(self._wait_time_ms)
            self._window_manager.process_events(keycode)
        
        # Cuando el bucle termina (p.ej., ESC presionado), liberar recursos.
        self._capture_manager.release()
        self._window_manager.destroy_window()

    
    def on_keypress(self, keycode):
        """
        Función callback que se llama cuando se presiona una tecla.
        Gestiona el cambio de filtros, capturas de pantalla y salida.
        """
        if keycode == 27:  # Tecla ESC
            # Señal para destruir la ventana y salir del bucle principal
            self._window_manager.destroy_window()
        elif keycode == ord('q'):  # Tecla 'q' (Filtro anterior)
            # Usa módulo (%) para ciclar la lista
            self._current_filter_index = (self._current_filter_index - 1) % len(self._filters)
            print(f"Cambiado a: {self._filters[self._current_filter_index].get_name()}")
        elif keycode == ord('e'):  # Tecla 'e' (Siguiente filtro)
            self._current_filter_index = (self._current_filter_index + 1) % len(self._filters)
            print(f"Cambiado a: {self._filters[self._current_filter_index].get_name()}")
        elif keycode == ord('s'):  # Tecla 's' (Guardar captura)
            self._screenshot_counter += 1
            filename = f'screenshot_{self._screenshot_counter:04d}.png'
            self._capture_manager.write_image(filename)
            print(f"Captura de pantalla guardada: {filename}")


if __name__ == '__main__':
    # Configurar el analizador de argumentos para requerir un archivo de video
    parser = argparse.ArgumentParser(description='Ejecuta la app con un archivo de video como entrada.')
    parser.add_argument('video_file', help='La ruta al archivo de video.')
    
    # Analizar los argumentos de la línea de comandos
    args = parser.parse_args()
    
    # Crear e iniciar la aplicación, pasando la ruta del video
    app = CameoApp(video_source=args.video_file)
    app.run()
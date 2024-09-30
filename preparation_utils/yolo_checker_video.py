#~ USAGE
#~ запускаем cmd от Администратора
#~ активируем виртуальную среду
# cd c:\my_campy
# cd d:\my_campy
# .\camenv8\Scripts\activate
# cd c:\my_campy\smart_city\preparation_utils
# cd d:\my_campy\smart_city\preparation_utils
#~~~~~~~~~~~~~~~~~~~~~~~~
# python yolo_checker_video.py
#~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ import the necessary packages
import os
from ultralytics import YOLO
import cv2

from task_timer import TaskTimer
from dirfile_worker import DirectoryFileWorker

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class YoloCheckerVideo:
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def __init__(self, src_fname1: str, yoloimg_wh1: int, weights_fname1: str, class_labels_lst1: list[str], feature_confidence1: float, rep_img_width2: int, rep_img_height2: int, dst_dir2: str):
    #~ засекаем время выполнения всего процесса
    self.timer_obj = TaskTimer()
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.src_fname1 = src_fname1
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.yoloimg_wh1 = yoloimg_wh1
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.weights_fname1 = weights_fname1
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.class_labels_lst1 = []
    for i in range(len(class_labels_lst1)):
      self.class_labels_lst1.append(class_labels_lst1[i])
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.feature_conf1 = feature_confidence1
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.rep_img_width2 = rep_img_width2
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.rep_img_height2 = rep_img_height2
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.dst_dir2 = dst_dir2
    #~~~~~~~~~~~~~~~~~~~~~~~~
    print(f'[INFO] self.src_fname1: `{self.src_fname1}`')
    print(f'[INFO] self.yoloimg_wh1: {self.yoloimg_wh1}')
    print(f'[INFO] self.weights_fname1: `{self.weights_fname1}`')
    print(f'[INFO] self.class_labels_lst1: len: {len(self.class_labels_lst1)}, {self.class_labels_lst1}')
    print(f'[INFO] self.feature_conf1: {round(self.feature_conf1,2)}')
    print(f'[INFO] self.rep_img_width2: {self.rep_img_width2}')
    print(f'[INFO] self.rep_img_height2: {self.rep_img_height2}')
    print(f'[INFO] self.dst_dir2: `{self.dst_dir2}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ объект для работы с файлами, папками и т.д.
    self.dir_filer = DirectoryFileWorker()
    self.dir_filer.remove_create_directory(self.dst_dir2)

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def video_check(self):
    print('~'*70)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ 0: "car-accident" - "ДТП - столкновение легковых автомобилей"
    #~ 1: "non-car-accident" - `не ДТП - столкновение легковых автомобилей"
    class_id = 0
    #~ цвет продетектированной сущности
    feature_color = (0, 0, 255)
    #~ цвет фона, чтобы лучше читалась подпись
    back_color = (255, 255, 255)
    #~ указываем размер изображения, на котором была обучена нейронка  
    #~ первое значение — это ширина,
    #~ второе значение — это высота.
    yotarget_size=(self.yoloimg_wh1, self.yoloimg_wh1)
    #~ размер кадра для отчетного документа
    reptarget_size=(self.rep_img_width2, self.rep_img_height2)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ открываю видео-файл
    vcam = cv2.VideoCapture(self.src_fname1)
    if not vcam.isOpened():
      print(f'[ERROR] can`t open video-file: `{self.src_fname1}`')
      return
    base_fname1,suffix_fname1 = self.dir_filer.get_fullfname_base_suffix(self.src_fname1)
    # print(f'[INFO]  base_fname1: `{base_fname1}`, suffix_fname1: `{suffix_fname1}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ оригинальные размеры кадра
    frame_width = int(vcam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'[INFO] original frame size: width: {frame_width}, height: {frame_height}, ratio: {round(frame_width/frame_height,5)}')
    print(f'[INFO] YOLO frame size: {self.yoloimg_wh1}')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ f - frame resize for report
    is_fresize = False
    if frame_width != self.rep_img_width2 or frame_height != self.rep_img_height2:
      is_fresize = True
    print(f'[INFO] frame resize for report: {is_fresize}')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    fps = vcam.get(cv2.CAP_PROP_FPS)
    print(f'[INFO] fps: {fps}')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ определяем кодек и FPS для сохранения видео
    #~~~~~~~~~~~~~~~~~~~~~~~~
    vid_fname2 = os.path.join(self.dst_dir2, base_fname1 + '_det.mp4')
    # print(f'[INFO] vid_fname2: {vid_fname2}')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vout = cv2.VideoWriter(vid_fname2, fourcc, fps, reptarget_size)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ YOLOv8 model on custom dataset
    #~ load a model
    #~ загружаем YOLO-модель с указанными весами
    #~~~~~~~~~~~~~~~~~~~~~~~~
    # model = YOLO('yolov8m.pt')
    model = YOLO(self.weights_fname1)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ побежали по кадрам -> детектируем на каждом события
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ число кадров с продетектированными сущностями -> счетчик кадров -> frame counter
    fcounter = 0
    digits = 5
    #~ frame total counter 
    ftcounter = 0
    while vcam.isOpened():
      print('~'*70)
      #~ читаем очередной кадр  
      ret, frame = vcam.read()
      ftcounter += 1
      if not ret:
        break
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ масштабирование/сжатие изображения до размеров 640x640
      #~ и предсказание модели
      #~ в любом случае сжимаем, так как видео-камер с разрешение 640x640 не бывает
      frame640 = cv2.resize(frame, yotarget_size, interpolation=cv2.INTER_AREA)
      #~ предсказание модели
      # results = model(frame640)[0]
      # yodets = model(frame640, imgsz=640, verbose=True)[0]
      yodets = model(frame640, imgsz=self.yoloimg_wh1, verbose=True)[0]
      # print(f'[INFO] yodets: `{yodets}`')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ f - feature frame
      # yodets_lst = yodets.boxes.data.tolist()
      # print(f'[INFO]  yodets_lst: len: {len(yodets)}')
      # for yodet in yodets_lst:
      #~~~~~~~~~~~~~~~~~~~~~~~~
      # if is_fresize:
      #~ для сохранения отчетного видео и кадров все кадры поджимаем
      frame = cv2.resize(frame, reptarget_size, interpolation=cv2.INTER_AREA)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      is_fdet = False
      for yodet in yodets.boxes.data.tolist():
        yox1, yoy1, yox2, yoy2, yoconf, yoclass_id = yodet
        # print(f'[INFO]  yox1: {yox1}, yoy1: {yoy1}, yox2: {yox2}, yoy2: {yoy2}')
        feature_id = int(yoclass_id)
        # print(f'[INFO]  yoclass_id: {yoclass_id}, class_id: {class_id}, feature_id: {feature_id}')
        if not feature_id == class_id:
          continue
        # print(f'[INFO]  yoconf: {yoconf}, self.feature_conf1: {self.feature_conf1}')
        if yoconf < self.feature_conf1:
          continue
        #~~~~~~~~~~~~~~~~~~~~~~~~
        #~ случилось детектирование необходимой сущности,
        #~ поэтому цикл поиска продетектированных сущностей в этом кадре останавливаю
        is_fdet = True
        # if is_fresize:
        #   frame = cv2.resize(frame, reptarget_size, interpolation=cv2.INTER_AREA)
        #~~~~~~~~~~~~~~~~~~~~~~~~
        #~ отрисовываем продетектированную сущность, если уверенность определения больше указанной  
        x_min = int(self.rep_img_width2*yox1/self.yoloimg_wh1)
        y_min = int(self.rep_img_height2*yoy1/self.yoloimg_wh1)
        x_max = int(self.rep_img_width2*yox2/self.yoloimg_wh1)
        y_max = int(self.rep_img_height2*yoy2/self.yoloimg_wh1)
        # print(f'[INFO]  x_min: {x_min}, y_min: {y_min}, x_max: {x_max}, y_max: {y_max}')
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), feature_color, 2)
        #~ и подписывае вероятность детектирования
        feature_lbl = f'{self.class_labels_lst1[class_id]}: {round(yoconf, 2)}'
        cv2.putText(frame, feature_lbl, (x_min+3, y_min-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, feature_color, 2)
        cv2.putText(frame, feature_lbl, (x_min+3, y_min-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, back_color, 1)
        #~~~~~~~~~~~~~~~~~~~~~~~~
        #~ детектирования одной сущности для alarm достаточно
        break
      #~~~~~~~~~~~~~~~~~~~~~~~~
      # print(f'[INFO]  is_fdet: {is_fdet}')
      if self.feature_conf1 < 0.3:
        #~ значит все кадры хотим проанализировать
        is_fdet = True
      if is_fdet:
        fcounter += 1
        #~ кадр отдельный сохраняем, только если была детекция
        # frame_fname2 = f'f{self.dir_filer.format_counter(fcounter, digits)}.jpg'
        frame_fname2 = f'f{self.dir_filer.format_counter(fcounter, digits)}.png'
        img_fname2 = os.path.join(self.dst_dir2, frame_fname2)
        print(f'[INFO]  img_fname2: `{img_fname2}`')
        cv2.imwrite(img_fname2, frame)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ и сохраняем кадр в видеофайл
      vout.write(frame)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ отображаем кадр
      cv2.imshow('car-accident-detection', frame)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ если нажата клавиша 'q', выходим из цикла
      # if cv2.waitKey(1) & 0xFF == ord('q'):
      #   break
      #~ если нажата клавиша 'esc', выходим из цикла
      key_press = cv2.waitKey(1) & 0xFF
      if 27 == key_press:
        print('[INFO] press key `esc` -> exit')
        break

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ освобождаем ресурсы
    #~~~~~~~~~~~~~~~~~~~~~~~~
    vout.release()
    vcam.release()
    #~ закрываем все окна
    cv2.destroyAllWindows()
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('='*70)
    print(f'[INFO] число обработанных видеокадров: {ftcounter}')
    print(f'[INFO] число видеокадров с продетектированными сущностями: {fcounter}')


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
  print('~'*70)
  print('[INFO] YoloCheckerVideo ver.2024.09.25')
  print('~'*70)
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ accident1_1280_720_25fps.mp4, feature_confidence1 = 0.92

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ директория с файлами-изображениями для детектирования объектов по рассчитаным весам в YOLO
  src_fname1 = 'c:/dataset_car_accident_detect/accident7_1920_1080_25fps.mp4'

  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ yolo image width,height ширина-высота изображения, на котором обучалась yolo
  yoloimg_wh1 = 640
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ файл с рассчитанными весами
  #~ 262 epochs completed in 1.626 hours.
  #~ Optimizer stripped from runs\detect\train4\weights\last.pt, 22.6MB
  #~ Optimizer stripped from runs\detect\train4\weights\best.pt, 22.5MB
  #~~~
  # weights_fname1 = 'c:/my_campy/smart_city/preparation_utils/runs/detect/train4/weights/last.pt'
  weights_fname1 = 'c:/my_campy/smart_city/preparation_utils/runs/detect/train4/weights/best.pt'
  #~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ список классов для детектирования
  class_labels_lst1 = ['car-accident', 'non-car-accident']
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ порог уверенности, не ниже которой будут продетектированы сущности
  #~ 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 0.92
  #~ 0.25, 0.85, 0.88, 0.92, 0.94
  feature_confidence1 = 0.88
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ rep - report
  rep_img_width2 = 960
  rep_img_height2 = 540
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ директория с файлами-изображениями результатами детектирования-отрисованными сущностями
  dst_dir2 = 'c:/dataset_car_accident_detect/20240918-epochs262/video7out_085'


  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~
  # python yolo_checker_video.py
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


  #~~~~~~~~~~~~~~~~~~~~~~~~
  ychrv_obj = YoloCheckerVideo(src_fname1, yoloimg_wh1, weights_fname1, class_labels_lst1, feature_confidence1, rep_img_width2, rep_img_height2, dst_dir2)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  ychrv_obj.video_check()
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ отображаем время выполнения программы
  ychrv_obj.timer_obj.elapsed_time()
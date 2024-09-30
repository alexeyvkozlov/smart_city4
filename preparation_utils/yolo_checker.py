#~ USAGE
#~ запускаем cmd от Администратора
#~ активируем виртуальную среду
# cd c:\my_campy
# cd d:\my_campy
# .\camenv8\Scripts\activate
# cd c:\my_campy\smart_city\preparation_utils
# cd d:\my_campy\smart_city\preparation_utils
#~~~~~~~~~~~~~~~~~~~~~~~~
# python yolo_checker.py
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
class YoloChecker:
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def __init__(self, src_dir1: str, yoloimg_wh1: int, weights_fname1: str, class_labels_lst1: list[str], feature_confidence1: float, rep_img_width2: int, rep_img_height2: int, dst_dir2: str):
    #~ засекаем время выполнения всего процесса
    self.timer_obj = TaskTimer()
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.src_dir1 = src_dir1
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.yoloimg_wh1 = yoloimg_wh1
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.weights_fname1 = weights_fname1
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.classes_lst1 = []
    for i in range(len(class_labels_lst1)):
      self.classes_lst1.append(class_labels_lst1[i])
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.feature_conf1 = feature_confidence1
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.rep_img_width2 = rep_img_width2
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.rep_img_height2 = rep_img_height2
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.dst_dir2 = dst_dir2
    #~~~~~~~~~~~~~~~~~~~~~~~~
    print(f'[INFO] self.src_dir1: `{self.src_dir1}`')
    print(f'[INFO] self.yoloimg_wh1: {self.yoloimg_wh1}')
    print(f'[INFO] self.weights_fname1: `{self.weights_fname1}`')
    print(f'[INFO] self.classes_lst1: len: {len(self.classes_lst1)}, {self.classes_lst1}')
    print(f'[INFO] self.feature_conf1: {round(self.feature_conf1,2)}')
    print(f'[INFO] self.rep_img_width2: {self.rep_img_width2}')
    print(f'[INFO] self.rep_img_height2: {self.rep_img_height2}')
    print(f'[INFO] self.dst_dir2: `{self.dst_dir2}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ объект для работы с файлами, папками и т.д.
    self.dir_filer = DirectoryFileWorker()
    self.dir_filer.remove_create_directory(self.dst_dir2)

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def image_check(self):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ 0: "car-accident" - "ДТП - столкновение легковых автомобилей"
    #~ 1: "non-car-accident" - `не ДТП - столкновение легковых автомобилей"
    class_id0 = 0
    class_id1 = 1
    #~ цвет продетектированной сущности
    feature_color0 = (0, 0, 255)
    feature_color1 = (255, 0, 0)
    #~ указываем размер изображения, на котором была обучена нейронка  
    #~ первое значение — это ширина,
    #~ второе значение — это высота.
    yotarget_size=(self.yoloimg_wh1, self.yoloimg_wh1)
    #~ размер кадра для отчетного документа
    reptarget_size=(self.rep_img_width2, self.rep_img_height2)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    img_lst = self.dir_filer.get_image_list(self.src_dir1)
    img_lst_len = len(img_lst)
    if img_lst_len < 1:
      print('[WARNING] img_lst is empty')
      return
    print(f'[INFO] img_lst: len: {img_lst_len}')
    # print(f'[INFO] img_lst: len: {img_lst_len}, `{img_lst}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ YOLOv8 model on custom dataset
    #~ load a model
    #~~~~~~~~~~~~~~~~~~~~~~~~
    # model = YOLO('yolov8m.pt')
    model = YOLO(self.weights_fname1)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ число кадров с продетектированными сущностями
    fcounter = 0
    #~ побежали по изображениям в списке
    for i in range(img_lst_len):
      print('~'*70)
      print(f'[INFO] {i}->{img_lst_len-1}: `{img_lst[i]}`')
      img_fname1 = os.path.join(self.src_dir1, img_lst[i])
      base_fname1,suffix_fname1 = self.dir_filer.get_fname_base_suffix(img_lst[i])
      # print(f'[INFO]  img_fname1: `{img_fname1}`')
      # print(f'[INFO]  base_fname1: `{base_fname1}`, suffix_fname1: `{suffix_fname1}`')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ проверяем изображение на корректность
      frame = cv2.imread(img_fname1)
      img_width1 = 0
      img_height1 = 0
      try:
        img_width1 = frame.shape[1]
        img_height1 = frame.shape[0]
      except:
        print(f'[WARNING] corrupted image: `{img_fname1}`')
        continue
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ масштабирование/сжатие изображения до размеров 640x640
      #~ и предсказание модели
      #~ в любом случае сжимаем, так как видео-камер с разрешение 640x640 не бывает
      #~ а изображение скорее всего с видеокамеры
      frame640 = cv2.resize(frame, yotarget_size, interpolation=cv2.INTER_AREA)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ предсказание модели
      # results = model(frame640)[0]
      # yodets = model(frame640, imgsz=640, verbose=True)[0]
      yodets = model(frame640, imgsz=self.yoloimg_wh1, verbose=True)[0]
      #~ f - feature frame
      #~ для сохранения отчетного видео и кадров все кадры поджимаем
      if img_width1 != self.rep_img_width2 or img_height1 != self.rep_img_height2:
        frame = cv2.resize(frame, reptarget_size, interpolation=cv2.INTER_AREA)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      is_fdet = False
      for yodet in yodets.boxes.data.tolist():
        yox1, yoy1, yox2, yoy2, yoconf, yoclass_id = yodet
        # print(f'[INFO]  yox1: {yox1}, yoy1: {yoy1}, yox2: {yox2}, yoy2: {yoy2}')
        feature_id = int(yoclass_id)
        # print(f'[INFO]  yoclass_id: {yoclass_id}, class_id: {class_id}, feature_id: {feature_id}')
        if not feature_id == class_id0:
          continue
        # print(f'[INFO]  yoconf: {yoconf}, self.feature_conf1: {self.feature_conf1}')
        if yoconf < self.feature_conf1:
          continue
        #~~~~~~~~~~~~~~~~~~~~~~~~
        if not is_fdet:
          is_fdet = True
        #~~~~~~~~~~~~~~~~~~~~~~~~
        #~ отрисовываем продетектированную сущность, если уверенность определения больше указанной  
        x_min = int(self.rep_img_width2*yox1/self.yoloimg_wh1)
        y_min = int(self.rep_img_height2*yoy1/self.yoloimg_wh1)
        x_max = int(self.rep_img_width2*yox2/self.yoloimg_wh1)
        y_max = int(self.rep_img_height2*yoy2/self.yoloimg_wh1)
        # print(f'[INFO]  x_min: {x_min}, y_min: {y_min}, x_max: {x_max}, y_max: {y_max}')
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), feature_color0, 2)
        # feature_lbl = f'{self.class_labels_lst1[class_id]}: {round(yoconf, 2)}'
        # cv2.putText(frame, feature_lbl, (x_min+3, y_min-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, feature_color, 2)
        # cv2.putText(frame, feature_lbl, (x_min+3, y_min-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      if is_fdet:
        fcounter += 1
        #~ и подписываю вероятность детектирования
        pred_lbl = f'predict: {self.classes_lst1[class_id0]}'
        feature_color = feature_color0
      else:
        pred_lbl = f'predict: {self.classes_lst1[class_id1]}'
        feature_color = feature_color1
      cv2.rectangle(frame, (0,0), (258,25), (255,255,255), -1)
      cv2.putText(frame, pred_lbl, (2,17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, feature_color, 1, cv2.LINE_AA)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ в любом случае сохраняем изображение с продетектированными сущностями,
      #~ даже если не было детекции
      # img_fname2 = os.path.join(self.dst_dir2, img_lst[i])
      img_fname2 = os.path.join(self.dst_dir2, base_fname1+'.png')
      # print(f'[INFO]  img_fname2: `{img_fname2}`')
      cv2.imwrite(img_fname2, frame)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ завершили чтение изображений по списку
    print('='*70)
    print(f'[INFO] число обработанных изображений: {img_lst_len}')
    print(f'[INFO] число изображений с продетектированными сущностями: {fcounter}')


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
  print('~'*70)
  print('[INFO] YoloChecker ver.2024.09.26')
  print('~'*70)
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ директория с файлами-изображениями для детектирования объектов по рассчитаным весам в YOLO
  src_dir1 = 'c:/my_campy/smart_city/smart_city_check_photo_video/car_accident/accident7-frames'
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
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ список классов для детектирования
  class_labels_lst1 = ['car-accident', 'non-car-accident']
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ порог уверенности, не ниже которой будут продетектированы сущности
  #~ 0.2, 0.3, 0.4, 0.5, 0.7
  feature_confidence1 = 0.25
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ rep - report
  rep_img_width2 = 960
  rep_img_height2 = 540
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ директория с файлами-изображениями результатами детектирования-отрисованными сущностями
  dst_dir2 = 'c:/my_campy/smart_city/smart_city_check_photo_video/car_accident/accident7-frames_out'


  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~
  # python yolo_checker.py
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


  #~~~~~~~~~~~~~~~~~~~~~~~~
  ychr_obj = YoloChecker(src_dir1, yoloimg_wh1, weights_fname1, class_labels_lst1, feature_confidence1, rep_img_width2, rep_img_height2, dst_dir2)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  ychr_obj.image_check()
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ отображаем время выполнения программы
  ychr_obj.timer_obj.elapsed_time()
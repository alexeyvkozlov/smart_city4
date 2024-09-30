#~ USAGE
#~ запускаем cmd от Администратора
#~ активируем виртуальную среду
# cd c:\my_campy
# cd d:\my_campy
# .\camenv8\Scripts\activate
# cd c:\my_campy\smart_city\preparation_utils
# cd d:\my_campy\smart_city\preparation_utils
#~~~~~~~~~~~~~~~~~~~~~~~~
# python yolo_checker_test.py
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
class YoloCheckerTest:
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def __init__(self, src_dir1: str, yoloimg_wh1: int, weights_fname1: str, class_labels_lst1: list[str], feature_confidence1: float, dst_dir2: str):
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
    self.dst_dir2 = dst_dir2
    #~~~~~~~~~~~~~~~~~~~~~~~~
    print(f'[INFO] self.src_dir1: `{self.src_dir1}`')
    print(f'[INFO] self.yoloimg_wh1: {self.yoloimg_wh1}')
    print(f'[INFO] self.weights_fname1: `{self.weights_fname1}`')
    print(f'[INFO] self.classes_lst1: len: {len(self.classes_lst1)}, {self.classes_lst1}')
    print(f'[INFO] self.feature_conf1: {round(self.feature_conf1,2)}')
    print(f'[INFO] self.dst_dir2: `{self.dst_dir2}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ объект для работы с файлами, папками и т.д.
    self.dir_filer = DirectoryFileWorker()
    self.dir_filer.remove_create_directory(self.dst_dir2)

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def get_class_inx(self, lbl_fname1: str) -> int:
    retVal = -1
    if not self.dir_filer.file_exists(lbl_fname1):
      return retVal
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ читаем файл по строкам
    lines1 = []
    input_file = open(lbl_fname1, 'r', encoding='utf-8')
    lines1 = input_file.readlines()
    input_file.close()
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ если вообще нет объектов или объект - фон, то класс -> 1
    retVal = 1
    #~ оставляем только не пустые строки
    for line1 in lines1:
      #~ удаляем пробелы в начале и конце строки
      line2 = line1.strip()
      if len(line2) < 1:
        continue
      #~~~~~~~~~~~~~~~~~~~~~~~~
      fields5 = line2.split()
      if not 5 == len(fields5):
        continue
      #~~~~~~~~~~~~~~~~~~~~~~~~
      try:
        #~ преобразование строки в числа
        class_id = int(fields5[0])
        x_center = float(fields5[1])
        y_center = float(fields5[2])
        width = float(fields5[3])
        height = float(fields5[4])
      except ValueError as e:
        print(f'[ERROR] произошла ошибка при преобразовании строки в число: `{line2}`, : {e}')
        continue
      #~~~~~~~~~~~~~~~~~~~~~~~~
      if 0 == class_id:
        retVal = 0
        break
    #~~~~~~~~~~~~~~~~~~~~~~~~
    return retVal

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
    print(f'[INFO] yotarget_size: {yotarget_size}')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    img_dir1 = os.path.join(self.src_dir1, 'images')
    lbl_dir1 = os.path.join(self.src_dir1, 'labels')
    print(f'[INFO] img_dir1: `{img_dir1}`')
    print(f'[INFO] lbl_dir1: `{lbl_dir1}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    img_lst = self.dir_filer.get_image_list(img_dir1)
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
    print(f'[INFO] self.weights_fname1: `{self.weights_fname1}`')
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
      img_fname1 = os.path.join(img_dir1, img_lst[i])
      lbl_fname1 = os.path.join(lbl_dir1, base_fname1 + '.txt')
      # print(f'[INFO]  img_fname1: `{img_fname1}`')
      class_inx = self.get_class_inx(lbl_fname1)
      # print(f'[INFO]  lbl_fname1: `{lbl_fname1}`, class_inx: {class_inx}')
      if -1 == class_inx:
        continue
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ проверяем изображение на корректность
      frame640 = cv2.imread(img_fname1)
      img_width1 = 0
      img_height1 = 0
      try:
        img_width1 = frame640.shape[1]
        img_height1 = frame640.shape[0]
      except:
        print(f'[WARNING] corrupted image: `{img_fname1}`')
        continue
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ масштабирование/сжатие изображения до размеров 640x640
      if img_width1 != self.yoloimg_wh1 or img_height1 != self.yoloimg_wh1:
        frame640 = cv2.resize(frame640, yotarget_size, interpolation=cv2.INTER_AREA)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ предсказание модели
      # results = model(frame640)[0]
      # yodets = model(frame640, imgsz=640, verbose=True)[0]
      yodets = model(frame640, imgsz=self.yoloimg_wh1, verbose=True)[0]
      #~ f - feature frame
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
        x_min = int(yox1)
        y_min = int(yoy1)
        x_max = int(yox2)
        y_max = int(yoy2)
        cv2.rectangle(frame640, (x_min, y_min), (x_max, y_max), feature_color0, 2)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      cv2.rectangle(frame640, (0,0), (262,48), (255,255,255), -1)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      class_lbl = f'original: {self.classes_lst1[class_inx]}'
      feature_color = feature_color0
      if 1 == class_inx:
        feature_color = feature_color1
      cv2.putText(frame640, class_lbl, (2,17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, feature_color, 1, cv2.LINE_AA)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      if is_fdet:
        fcounter += 1
        #~ и подписываю вероятность детектирования
        pred_lbl = f'predict: {self.classes_lst1[class_id0]}'
        feature_color = feature_color0
      else:
        pred_lbl = f'predict: {self.classes_lst1[class_id1]}'
        feature_color = feature_color1
      cv2.putText(frame640, pred_lbl, (2,40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, feature_color, 1, cv2.LINE_AA)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ сохраняем изображение
      img_fname2 = os.path.join(self.dst_dir2, base_fname1+'.png')
      print(f'[INFO]  img_fname2: `{img_fname2}`')
      cv2.imwrite(img_fname2, frame640)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ завершили чтение изображений по списку
    print('='*70)
    print(f'[INFO] число обработанных изображений: {img_lst_len}')
    print(f'[INFO] число изображений с продетектированными сущностями: {fcounter}')


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
  print('~'*70)
  print('[INFO] YoloCheckerTest ver.2024.09.26')
  print('~'*70)
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ директория с файлами-изображениями для детектирования объектов по рассчитаным весам в YOLO
  src_dir1 = 'c:/my_campy/smart_city/smart_city_check_photo_video/car_accident/dataset_car_accident/test'
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
  #~ директория с файлами-изображениями результатами детектирования-отрисованными сущностями
  dst_dir2 = 'c:/my_campy/smart_city/smart_city_check_photo_video/car_accident/dataset_car_accident_test_out'


  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~
  # python yolo_checker_test.py
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


  #~~~~~~~~~~~~~~~~~~~~~~~~
  ychr_obj = YoloCheckerTest(src_dir1, yoloimg_wh1, weights_fname1, class_labels_lst1, feature_confidence1, dst_dir2)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  ychr_obj.image_check()
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ отображаем время выполнения программы
  ychr_obj.timer_obj.elapsed_time()
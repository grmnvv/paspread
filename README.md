# RUPassportRead
## Библиотека для чтения данных с паспорта РФ.
RUPassportRead – это модуль по распознаванию данных из изображения паспорта, написанный на языке программирования Python, на основе таких библиотек, как: OpenCV, Tesseract OCR.

Основная задача данного проекта стояла в получении данных пользователя из фотографии паспорта. Данная процедура была осуществлена через поиск и получение информации с Машино-Читаемой Зоны (Далее mrz). 

Процедура распознавания может быть довольно медленной - около 10 или более секунд для некоторых документов. Его точность не идеальна, но, по-видимому, прилична с точки зрения тестовых документов, доступных мне - примерно в 80% случаев, когда на странице есть четко видимый MRZ, система распознает его и извлекает текст в меру возможностей базового механизма распознавания текста. (Google Tesseract).

Для использования понадобится `Python версии 3.7 – 3.9`.

Самый простой способ установить пакет - через pip:

` pip install rupasportread`

Обратите внимание, что rupasportread зависит, среди прочего, от numpy, imutils, и OpenCV. Установка этих библиотек, хотя и автоматическая, но  может занять некоторое время или иногда завершаться сбоем по разным причинам. Если это произойдет, попробуйте установить напрямую с помощью команд pip. Другим удобным вариантом является использование дистрибутива Python с предварительно установленными основными пакетами (Anaconda Python на данный момент является отличным выбором).

Кроме того, у вас должен быть установлен Tesseract OCR и добавлен в системный путь: инструмент tesseract должен быть доступен из командной строки. Загрузить можно по этой [ссылке](https://github.com/UB-Mannheim/tesseract/wiki). 

Найдите установщик «tesseract-ocr-w64-setup-v5.0.1.20220118.exe». И при установке выберите пакеты с Английским языком / Латиницей. После установки добавьте путь к файлу в переменных средах для системы. Добавьте в Path - `C:\Program Files\Tesseract-OCR` (стандартный путь, если ничего не менять при установке).

Поддерживаются форматы изображений: bmp, jpeg/jpg, png

Использование реализуется через добавление 2х строчек кода в ваш проект Python:

```python
import rupasportread as pr 

pr.catching('Путь к вашему файлу')
```



Ответ будет получен в формате словаря pasdata. Обратиться к нему можно с помощью ключей:
1.	Surname – фамилия 
2.	Name – имя 
3.	Mid – отчество 
4.	Date – дата рождения
5.	Series – серия паспорта
6.	Number – номер паспорта 

Также есть возможность скачать кадрированный документ, или если есть потребность, то можно использовать, как получение обрезанной фотографии листа/книги/паспорта или любого другого предмета. Реализуется через функцию:
```python
import rupasportread as pr

pr.download('путь к изображению', 'название изображения . расширение (jpg/jpeg/png)')
```

Обрезанное изображение будет сохранено в папку, где лежит файл с вашим кодом.

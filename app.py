import telebot
from telebot import types
from bot_description.texts import *
from blindness_detection.model import MakePredict

import os
import time

from flask import Flask, request
from flask import render_template


from PIL import Image
from io import BytesIO
import numpy as np

## Constants
TOKEN = '841484068:AAFciZH0o5mT8Zo_r7upTTGt-BdZ_Y0oiJk'
STICKER_ID = 'CAADAgADAQAD3BQ9Js2i8jeh-Q6nAg'
GIF_ID = 'CgADAgAD4AMAAtmwSUmif7hi8FXP3gI'

bot = telebot.TeleBot(TOKEN)
app = Flask(__name__)

## Bot menu
markup_menu = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
btn_service = types.KeyboardButton('About Service')
btn_ethos = types.KeyboardButton('Our Ethos')
btn_apm = types.KeyboardButton('About Prediction Algorithms')

markup_menu.add(btn_service, btn_apm, btn_ethos)

## Command handlers
@bot.message_handler(commands=['start'])
def send_welcome(message):
    user_first_name = message.from_user.first_name
    bot.reply_to(message, f"Welcome {user_first_name}, I'm a bot-ophthalmologist, you can upload a " \
           "snapshot of the retinal fundus, and i will make a prediction the " \
           "presence of diabetic retinopathy in the picture on a scale of 0 to 4. \n"\
           "Update: Now I can segment the retinal blood vessels.",
                 reply_markup=markup_menu)


@bot.message_handler(commands=['help'])
def send_welcome(message):
    bot.reply_to(message, HELP_DISCRIPT,
                 reply_markup=markup_menu)


@bot.message_handler(commands=['list'])
def send_welcome(message):
    bot.reply_to(message, LIST_DISCRIPT,
                 reply_markup=markup_menu)


@bot.message_handler(commands=['contact'])
def send_welcome(message):
    bot.reply_to(message, CONTACT_DISCRIPT,
                 reply_markup=markup_menu)



## Information buttons
@bot.message_handler(func=lambda message: True)
def echo_all(message):
    if message.text == 'About Service':
        bot.reply_to(message, ABOUT_SERVICE,
                     reply_markup=markup_menu)

    elif message.text == 'About Prediction Algorithms':
        bot.reply_to(message, ABOUT_PREDICTION_MODEL,
                     reply_markup=markup_menu)

    elif message.text == 'Our Ethos':
        bot.reply_to(message, OUT_ETHOS,
                     reply_markup=markup_menu)

## File handlers
@bot.message_handler(content_types=['sticker'])
def sticker_handler(message):
    bot.send_sticker(message.chat.id, STICKER_ID)

@bot.message_handler(content_types=['document'])
def gif_handler(message):
    bot.send_document(message.chat.id, GIF_ID)
    bot.send_message(message.chat.id, FILE_DISCRIPT)

# Photo handler for make prediction
@bot.message_handler(content_types=['photo'])
def send_prediction_on_photo(message):
    print("Start working on photo")
    # get photo id and upload it into memory
    # [-1] index corresponds to the best quality
    photo_id = message.photo[-1].file_id
    photo_info = bot.get_file(photo_id)
    photo_bytes = bot.download_file(photo_info.file_path)
    bot.send_message(message.chat.id, 'Your photo is in line, please wait.',
                     reply_markup=markup_menu)

    # create BytesIO wrapper for the image
    img = Image.open(BytesIO(photo_bytes))
    img = img.resize((312, 312), Image.BILINEAR)
    prob, label, heatmap, seg_prediction = MakePredict().make_predict(img)


    # send prediction with probability
    prob = np.array(prob[label]) * 100
    prob = np.around(prob, 2)                
    bot.send_message(message.chat.id, f'This is class {str(label)} with probability {str(prob) + " %"}',
                     reply_markup=markup_menu)

    # send alpha heatmap
    bot.send_message(message.chat.id, "Visual explanations from neural net via gradient-based localization",
                     reply_markup=markup_menu)
    
    stream = BytesIO()
    heatmap.save(stream, format='PNG')
    stream.flush()
    stream.seek(0)
    bot.send_photo(message.chat.id, stream)
    time.sleep(1)
    print("Sent First Photo To User")

    time.sleep(1)
  
    # send a mask of blood vessels
    bot.send_message(message.chat.id, "Retinal blood vessels segmentation",
                     reply_markup=markup_menu)
 
    stream = BytesIO()
    seg_prediction.save(stream, format='PNG')
    stream.flush()
    stream.seek(0)
    bot.send_photo(message.chat.id, stream)
    time.sleep(1)
    print("Sent Second Photo To User")
    seg_prediction.save(stream, format='PNG')



@app.route('/' + TOKEN, methods=['POST'])
def getMessage():
    bot.process_new_updates([telebot.types.Update.de_json(request.stream.read().decode("utf-8"))])
    return "!", 200


@app.route("/")
def webhook():
    bot.remove_webhook()
    bot.set_webhook(url='https://eyemedservice.herokuapp.com/' + TOKEN)
    return render_template("index.html"), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5000)))


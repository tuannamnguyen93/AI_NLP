# -*- coding:utf8 -*-
# !/usr/bin/env python
from __future__ import print_function
from future.standard_library import install_aliases
import json
from flask import Flask, jsonify, request, make_response
from flask_cors import CORS, cross_origin
import os
from flask import Flask
from Intent import IntentClassifier
from Entity import EntityClassifier
from models.extras import normalize_text
install_aliases()
app = Flask(__name__)
cors = CORS(app)

threshold_confidence = 0.55
@app.errorhandler(400)
def not_found(error):
    return make_response(jsonify({'error': 'Bad request! Thiếu thông tin'}), 400)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.route('/train', methods=['POST'])
@cross_origin()

def train():
    req = request.get_json(silent=True, force=True)
    res = processTrain(req)
    res = json.dumps(res, indent=4)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r


def processTrain(req):
    string_req = req["botId"]
    arr = string_req.split("_")
    botId = arr[1]
    type_intent = req["type"]
    # print('type = '+type_intent)
    print('BOT ID = '+botId)
    intent_class = IntentClassifier(botId, type_intent)
    # entity_class = EntityClassifier(botId)
    intent_class.trainmodel(botId)
    # entity_class.train_entity_model()
    response = {'querry': 'Train done!','BotID':botId }
    return response


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5558))
    print("Starting app on port %d" % port)
    app.run(debug=False, port=port,host = '0.0.0.0')
    # req = {
    #         "botId":"2",
    #         }
    # processTrain(req)
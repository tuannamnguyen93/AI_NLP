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

threshold_confidence = 0.6
@app.errorhandler(400)
def not_found(error):
    return make_response(jsonify({'error': 'Bad request! Thiếu thông tin'}), 400)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)
@app.errorhandler(500)
def not_found(error):
    return make_response(jsonify({'error': 'Some thing wrong with models'}), 500)

@app.route('/conversation', methods=['POST'])
@cross_origin()
def conversation():
    req = request.get_json(silent=True, force=True)
    res = processRequest(req)
    res = json.dumps(res, indent=4)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r

def processRequest(req):
    query = req["query"]
    botId = req["botId"]
    type = req["type"]
    query = normalize_text(query)
    sessionId = req["sessionId"]
    Intent_Class = IntentClassifier(botId, type)
    try:
        intent = Intent_Class.classify_intent(query,botId,threshold_confidence=threshold_confidence)
    except ValueError:
        intent = ['Error',0,'Error']

    entity_class = EntityClassifier(botId)
    try:
        entities = entity_class.ner(query)
    except Exception as err:
        entities = [['Error','Error'],]
        print(err)
    intentname,confidence,response = intent[0], intent[1],intent[2]
    response = {
        'sessionId': sessionId,
        'resolvedQuery': query,
        'intentName':intentname,
        'confidence': confidence,
        'response':  [{"speech": response},],
        # 'entities':entities
    }

    list_entities,my_list = [],[]
    for entity in entities:
        if entity[1] not in list_entities:
            list_entities.append(entity[1])
    for my_entity in list_entities:
        dict_temp = {}
        temp_list = []
        for entity in entities:
            if entity[1]==my_entity:
                temp_list.append(entity[0])
        dict_temp[my_entity] = temp_list
        my_list.append(dict_temp)

    response['entities'] = my_list
    return response


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5557))
    print("Starting app on port %d" % port)
    app.run(debug=False, port=port,host = '0.0.0.0')

    # req = {
    #         "botId":"2",
    #         "query":u"Ở Xuân Thủy Cầu giấy Hà Nội hoang hoa tham, thì mua thuốc panaldol vĩnh phúc phú thọ ở trang phuc linh Hà Nội chỗ nào",
    #
    #         "sessionId":"123"
    #         }
    # # req = json.dumps(req,str = 'utf8')
    # import json
    # print(json.dumps(processRequest(req),indent=4))
